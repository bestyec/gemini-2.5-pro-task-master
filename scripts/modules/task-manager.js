/**
 * task-manager.js
 * Task management functions for the Task Master CLI
 */

import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import boxen from 'boxen';
import Table from 'cli-table3';
import readline from 'readline';
import { Anthropic } from '@anthropic-ai/sdk';

import { 
  CONFIG, 
  log, 
  readJSON, 
  writeJSON, 
  sanitizePrompt,
  findTaskById,
  readComplexityReport,
  findTaskInComplexityReport,
  truncate
} from './utils.js';

import {
  displayBanner,
  getStatusWithColor,
  formatDependenciesWithStatus,
  getComplexityWithColor,
  startLoadingIndicator,
  stopLoadingIndicator,
  createProgressBar
} from './ui.js';

import {
  callGenerativeAI,
  generateSubtasks,
  generateSubtasksWithPerplexity,
  analyzeTaskComplexity,
  callGenerativeAIForUpdate, // Added import for the new update function
  getPerplexityClient
} from './ai-services.js';

import {
  validateTaskDependencies,
  validateAndFixDependencies
} from './dependency-manager.js';

// Initialize Anthropic client
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// Import perplexity if available
let perplexity;

try {
  if (process.env.PERPLEXITY_API_KEY) {
    // Using the existing approach from ai-services.js
    const OpenAI = (await import('openai')).default;
    
    perplexity = new OpenAI({
      apiKey: process.env.PERPLEXITY_API_KEY, 
      baseURL: 'https://api.perplexity.ai',
    });
    
    log('info', `Initialized Perplexity client with OpenAI compatibility layer`);
  }
} catch (error) {
  log('warn', `Failed to initialize Perplexity client: ${error.message}`);
  log('warn', 'Research-backed features will not be available');
}

/**
 * Parse a PRD file and generate tasks
 * @param {string} prdPath - Path to the PRD file
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {number} numTasks - Number of tasks to generate
 */
async function parsePRD(prdPath, tasksPath, numTasks) {
  try {
    log('info', `Parsing PRD file: ${prdPath}`);
    
    // Read the PRD content
    const prdContent = fs.readFileSync(prdPath, 'utf8');
    
    // Call the updated AI function
    const tasksData = await callGenerativeAI(prdContent, prdPath, numTasks);
    
    // Create the directory if it doesn't exist
    const tasksDir = path.dirname(tasksPath);
    if (!fs.existsSync(tasksDir)) {
      fs.mkdirSync(tasksDir, { recursive: true });
    }
    
    // Write the tasks to the file
    writeJSON(tasksPath, tasksData);
    
    log('success', `Successfully generated ${tasksData.tasks.length} tasks from PRD`);
    log('info', `Tasks saved to: ${tasksPath}`);
    
    // Generate individual task files
    await generateTaskFiles(tasksPath, tasksDir);
    
    console.log(boxen(
      chalk.green(`Successfully generated ${tasksData.tasks.length} tasks from PRD`),
      { padding: 1, borderColor: 'green', borderStyle: 'round' }
    ));
    
    console.log(boxen(
      chalk.white.bold('Next Steps:') + '\n\n' +
      `${chalk.cyan('1.')} Run ${chalk.yellow('task-master list')} to view all tasks\n` +
      `${chalk.cyan('2.')} Run ${chalk.yellow('task-master expand --id=<id>')} to break down a task into subtasks`,
      { padding: 1, borderColor: 'cyan', borderStyle: 'round', margin: { top: 1 } }
    ));
  } catch (error) {
    log('error', `Error parsing PRD: ${error.message}`);
    console.error(chalk.red(`Error: ${error.message}`));
    
    if (CONFIG.debug) {
      console.error(error);
    }
    
    process.exit(1);
  }
}

/**
 * Update tasks based on new context
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {number} fromId - Task ID to start updating from
 * @param {string} prompt - Prompt with new context
 * @param {boolean} useResearch - Whether to use Perplexity AI for research
 */
async function updateTasks(tasksPath, fromId, prompt, useResearch = false) {
  try {
    log('info', `Updating tasks from ID ${fromId} with prompt: "${prompt}"`);
    
    // Validate research flag
    if (useResearch && (!perplexity || !process.env.PERPLEXITY_API_KEY)) {
      log('warn', 'Perplexity AI is not available. Falling back to Claude AI.');
      console.log(chalk.yellow('Perplexity AI is not available (API key may be missing). Falling back to Claude AI.'));
      useResearch = false;
    }
    
    // Read the tasks file
    const data = readJSON(tasksPath);
    if (!data || !data.tasks) {
      throw new Error(`No valid tasks found in ${tasksPath}`);
    }
    
    // Find tasks to update (ID >= fromId and not 'done')
    const tasksToUpdate = data.tasks.filter(task => task.id >= fromId && task.status !== 'done');
    if (tasksToUpdate.length === 0) {
      log('info', `No tasks to update (all tasks with ID >= ${fromId} are already marked as done)`);
      console.log(chalk.yellow(`No tasks to update (all tasks with ID >= ${fromId} are already marked as done)`));
      return;
    }
    
    // Show the tasks that will be updated
    const table = new Table({
      head: [
        chalk.cyan.bold('ID'),
        chalk.cyan.bold('Title'),
        chalk.cyan.bold('Status')
      ],
      colWidths: [5, 60, 10]
    });
    
    tasksToUpdate.forEach(task => {
      table.push([
        task.id,
        truncate(task.title, 57),
        getStatusWithColor(task.status)
      ]);
    });
    
    console.log(boxen(
      chalk.white.bold(`Updating ${tasksToUpdate.length} tasks`),
      { padding: 1, borderColor: 'blue', borderStyle: 'round', margin: { top: 1, bottom: 0 } }
    ));
    
    console.log(table.toString());
    
    // Build the system prompt (Context for the AI)
    const systemPrompt = `You are an AI assistant helping to update software development tasks based on new context.
You will be given a set of existing tasks and a prompt describing changes or new implementation details.
Your job is to update the tasks to reflect these changes, while preserving their basic structure as much as possible.

Guidelines:
1. Maintain the original task IDs, statuses, dependencies, and priorities unless the new context explicitly dictates a change.
2. Focus on updating titles, descriptions, details, and test strategies to align with the new information.
3. Only modify what is necessary based on the provided prompt.
4. Return the complete list of updated tasks in the same order they were provided.
5. Ensure the output is ONLY a valid JSON array containing the updated task objects.`;

    const taskData = JSON.stringify(tasksToUpdate, null, 2);
    
    let updatedTasks;
    const loadingIndicator = startLoadingIndicator(useResearch 
      ? 'Updating tasks with Perplexity AI research...' 
      : 'Updating tasks with Generative AI...');
    
    try {
      if (useResearch) {
        log('info', 'Using Perplexity AI for research-backed task updates');
        
        // Ensure Perplexity client is fetched correctly if needed here
        const perplexityClient = getPerplexityClient(); // Use the imported getter
        
        // Call Perplexity AI using format consistent with ai-services.js
        const perplexityModel = process.env.PERPLEXITY_MODEL || 'sonar-pro'; // Use sonar-pro as default
        const result = await perplexityClient.chat.completions.create({ // Use the client instance
          model: perplexityModel,
          messages: [
            {
              role: "system", 
              content: `${systemPrompt}\n\nAdditionally, please research the latest best practices and technical considerations related to the user's update prompt.`
            },
            {
              role: "user",
              content: `Here are the tasks to update:
${taskData}

Please update these tasks based on the following new context:
${prompt}

Return only the updated tasks as a valid JSON array.`
            }
          ],
          temperature: 0.2, // Lower temperature for more direct updates
          max_tokens: CONFIG.maxTokens,
        });
        
        const responseText = result?.choices?.[0]?.message?.content || '[]'; // Default to empty array string
        // Basic parsing for Perplexity response (assuming it returns JSON array)
        try {
            let cleanedText = responseText.trim();
            const jsonMatch = cleanedText.match(/```(?:json)?\s*(\[\s\S]*?\])\s*```/s) || cleanedText.match(/(\[\s\S]*?\])/s);
            if (jsonMatch && jsonMatch[1]) cleanedText = jsonMatch[1];
            updatedTasks = JSON.parse(cleanedText);
        } catch (parseError) {
             log('error', `Failed to parse Perplexity update response: ${parseError.message}`);
             log('debug', `Raw Perplexity response: ${responseText}`);
             updatedTasks = tasksToUpdate; // Fallback to original tasks on parse error
        }
      } else {
        log('info', 'Using Generative AI for task updates'); // Updated text
        // Call the actual Generative AI update function
        updatedTasks = await callGenerativeAIForUpdate(tasksToUpdate, prompt, systemPrompt);
      }
      
      // Replace the tasks in the original data
      updatedTasks.forEach(updatedTask => {
        const index = data.tasks.findIndex(t => t.id === updatedTask.id);
        if (index !== -1) {
          data.tasks[index] = updatedTask;
        }
      });
      
      // Write the updated tasks to the file
      writeJSON(tasksPath, data);
      
      log('success', `Successfully updated ${updatedTasks.length} tasks`);
      
      // Generate individual task files
      await generateTaskFiles(tasksPath, path.dirname(tasksPath));
      
      console.log(boxen(
        chalk.green(`Successfully updated ${updatedTasks.length} tasks`),
        { padding: 1, borderColor: 'green', borderStyle: 'round' }
      ));
    } finally {
      stopLoadingIndicator(loadingIndicator);
    }
  } catch (error) {
    log('error', `Error updating tasks: ${error.message}`);
    console.error(chalk.red(`Error: ${error.message}`));
    
    if (CONFIG.debug) {
      console.error(error);
    }
    
    process.exit(1);
  }
}

/**
 * Generate individual task files from tasks.json
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {string} outputDir - Output directory for task files
 */
function generateTaskFiles(tasksPath, outputDir) {
  try {
    log('info', `Reading tasks from ${tasksPath}...`);
    const data = readJSON(tasksPath);
    if (!data || !data.tasks) {
      throw new Error(`No valid tasks found in ${tasksPath}`);
    }
    
    // Create the output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    log('info', `Found ${data.tasks.length} tasks to generate files for.`);
    
    // Validate and fix dependencies before generating files
    log('info', `Validating and fixing dependencies before generating files...`);
    validateAndFixDependencies(data, tasksPath);
    
    // Generate task files
    log('info', 'Generating individual task files...');
    data.tasks.forEach(task => {
      const taskPath = path.join(outputDir, `task_${task.id.toString().padStart(3, '0')}.txt`);
      
      // Format the content
      let content = `# Task ID: ${task.id}\n`;
      content += `# Title: ${task.title}\n`;
      content += `# Status: ${task.status || 'pending'}\n`;
      
      // Format dependencies with their status
      if (task.dependencies && task.dependencies.length > 0) {
        content += `# Dependencies: ${formatDependenciesWithStatus(task.dependencies, data.tasks, false)}\n`;
      } else {
        content += '# Dependencies: None\n';
      }
      
      content += `# Priority: ${task.priority || 'medium'}\n`;
      content += `# Description: ${task.description || ''}\n`;
      
      // Add more detailed sections
      content += '# Details:\n';
      content += (task.details || '').split('\n').map(line => line).join('\n');
      content += '\n\n';
      
      content += '# Test Strategy:\n';
      content += (task.testStrategy || '').split('\n').map(line => line).join('\n');
      content += '\n';
      
      // Add subtasks if they exist
      if (task.subtasks && task.subtasks.length > 0) {
        content += '\n# Subtasks:\n';
        
        task.subtasks.forEach(subtask => {
          content += `## ${subtask.id}. ${subtask.title} [${subtask.status || 'pending'}]\n`;
          
          if (subtask.dependencies && subtask.dependencies.length > 0) {
            // Format subtask dependencies
            let subtaskDeps = subtask.dependencies.map(depId => {
              if (typeof depId === 'number') {
                // Handle numeric dependencies to other subtasks
                const foundSubtask = task.subtasks.find(st => st.id === depId);
                if (foundSubtask) {
                  // Just return the plain ID format without any color formatting
                  return `${task.id}.${depId}`;
                }
              }
              return depId.toString();
            }).join(', ');
            
            content += `### Dependencies: ${subtaskDeps}\n`;
          } else {
            content += '### Dependencies: None\n';
          }
          
          content += `### Description: ${subtask.description || ''}\n`;
          content += '### Details:\n';
          content += (subtask.details || '').split('\n').map(line => line).join('\n');
          content += '\n\n';
        });
      }
      
      // Write the file
      fs.writeFileSync(taskPath, content);
      log('info', `Generated: task_${task.id.toString().padStart(3, '0')}.txt`);
    });
    
    log('success', `All ${data.tasks.length} tasks have been generated into '${outputDir}'.`);
  } catch (error) {
    log('error', `Error generating task files: ${error.message}`);
    console.error(chalk.red(`Error generating task files: ${error.message}`));
    
    if (CONFIG.debug) {
      console.error(error);
    }
    
    process.exit(1);
  }
}

/**
 * Set the status of a task
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {string} taskIdInput - Task ID(s) to update
 * @param {string} newStatus - New status
 */
async function setTaskStatus(tasksPath, taskIdInput, newStatus) {
  try {
    displayBanner();
    
    console.log(boxen(
      chalk.white.bold(`Updating Task Status to: ${newStatus}`),
      { padding: 1, borderColor: 'blue', borderStyle: 'round' }
    ));
    
    log('info', `Reading tasks from ${tasksPath}...`);
    const data = readJSON(tasksPath);
    if (!data || !data.tasks) {
      throw new Error(`No valid tasks found in ${tasksPath}`);
    }
    
    // Handle multiple task IDs (comma-separated)
    const taskIds = taskIdInput.split(',').map(id => id.trim());
    const updatedTasks = [];
    
    // Update each task
    for (const id of taskIds) {
      await updateSingleTaskStatus(tasksPath, id, newStatus, data);
      updatedTasks.push(id);
    }
    
    // Write the updated tasks to the file
    writeJSON(tasksPath, data);
    
    // Validate dependencies after status update
    log('info', 'Validating dependencies after status update...');
    validateTaskDependencies(data.tasks);
    
    // Generate individual task files
    log('info', 'Regenerating task files...');
    await generateTaskFiles(tasksPath, path.dirname(tasksPath));
    
    // Display success message
    for (const id of updatedTasks) {
      const task = findTaskById(data.tasks, id);
      const taskName = task ? task.title : id;
      
      console.log(boxen(
        chalk.white.bold(`Successfully updated task ${id} status:`) + '\n' +
        `From: ${chalk.yellow(task ? task.status : 'unknown')}\n` +
        `To:   ${chalk.green(newStatus)}`,
        { padding: 1, borderColor: 'green', borderStyle: 'round' }
      ));
    }
  } catch (error) {
    log('error', `Error setting task status: ${error.message}`);
    console.error(chalk.red(`Error: ${error.message}`));
    
    if (CONFIG.debug) {
      console.error(error);
    }
    
    process.exit(1);
  }
}

/**
 * Update the status of a single task
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {string} taskIdInput - Task ID to update
 * @param {string} newStatus - New status
 * @param {Object} data - Tasks data
 */
async function updateSingleTaskStatus(tasksPath, taskIdInput, newStatus, data) {
  // Check if it's a subtask (e.g., "1.2")
  if (taskIdInput.includes('.')) {
    const [parentId, subtaskId] = taskIdInput.split('.').map(id => parseInt(id, 10));
    
    // Find the parent task
    const parentTask = data.tasks.find(t => t.id === parentId);
    if (!parentTask) {
      throw new Error(`Parent task ${parentId} not found`);
    }
    
    // Find the subtask
    if (!parentTask.subtasks) {
      throw new Error(`Parent task ${parentId} has no subtasks`);
    }
    
    const subtask = parentTask.subtasks.find(st => st.id === subtaskId);
    if (!subtask) {
      throw new Error(`Subtask ${subtaskId} not found in parent task ${parentId}`);
    }
    
    // Update the subtask status
    const oldStatus = subtask.status || 'pending';
    subtask.status = newStatus;
    
    log('info', `Updated subtask ${parentId}.${subtaskId} status from '${oldStatus}' to '${newStatus}'`);
    
    // Check if all subtasks are done (if setting to 'done')
    if (newStatus.toLowerCase() === 'done' || newStatus.toLowerCase() === 'completed') {
      const allSubtasksDone = parentTask.subtasks.every(st => 
        st.status === 'done' || st.status === 'completed');
      
      // Suggest updating parent task if all subtasks are done
      if (allSubtasksDone && parentTask.status !== 'done' && parentTask.status !== 'completed') {
        console.log(chalk.yellow(`All subtasks of parent task ${parentId} are now marked as done.`));
        console.log(chalk.yellow(`Consider updating the parent task status with: task-master set-status --id=${parentId} --status=done`));
      }
    }
  } else {
    // Handle regular task
    const taskId = parseInt(taskIdInput, 10);
    const task = data.tasks.find(t => t.id === taskId);
    
    if (!task) {
      throw new Error(`Task ${taskId} not found`);
    }
    
    // Update the task status
    const oldStatus = task.status || 'pending';
    task.status = newStatus;
    
    log('info', `Updated task ${taskId} status from '${oldStatus}' to '${newStatus}'`);
    
    // If marking as done, also mark all subtasks as done
    if ((newStatus.toLowerCase() === 'done' || newStatus.toLowerCase() === 'completed') && 
        task.subtasks && task.subtasks.length > 0) {
      
      const pendingSubtasks = task.subtasks.filter(st => 
        st.status !== 'done' && st.status !== 'completed');
      
      if (pendingSubtasks.length > 0) {
        log('info', `Also marking ${pendingSubtasks.length} subtasks as '${newStatus}'`);
        
        pendingSubtasks.forEach(subtask => {
          subtask.status = newStatus;
        });
      }
    }
  }
}

/**
 * List all tasks
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {string} statusFilter - Filter by status
 * @param {boolean} withSubtasks - Whether to show subtasks
 */
function listTasks(tasksPath, statusFilter, withSubtasks = false) {
  try {
    displayBanner();
    const data = readJSON(tasksPath);
    if (!data || !data.tasks) {
      throw new Error(`No valid tasks found in ${tasksPath}`);
    }
    
    // Filter tasks by status if specified
    const filteredTasks = statusFilter 
      ? data.tasks.filter(task => 
          task.status && task.status.toLowerCase() === statusFilter.toLowerCase())
      : data.tasks;
    
    // Calculate completion statistics
    const totalTasks = data.tasks.length;
    const completedTasks = data.tasks.filter(task => 
      task.status === 'done' || task.status === 'completed').length;
    const completionPercentage = totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0;
    
    // Count statuses
    const doneCount = completedTasks;
    const inProgressCount = data.tasks.filter(task => task.status === 'in-progress').length;
    const pendingCount = data.tasks.filter(task => task.status === 'pending').length;
    const blockedCount = data.tasks.filter(task => task.status === 'blocked').length;
    const deferredCount = data.tasks.filter(task => task.status === 'deferred').length;
    
    // Count subtasks
    let totalSubtasks = 0;
    let completedSubtasks = 0;
    
    data.tasks.forEach(task => {
      if (task.subtasks && task.subtasks.length > 0) {
        totalSubtasks += task.subtasks.length;
        completedSubtasks += task.subtasks.filter(st => 
          st.status === 'done' || st.status === 'completed').length;
      }
    });
    
    const subtaskCompletionPercentage = totalSubtasks > 0 ? 
      (completedSubtasks / totalSubtasks) * 100 : 0;
      
    // Create progress bars
    const taskProgressBar = createProgressBar(completionPercentage, 30);
    const subtaskProgressBar = createProgressBar(subtaskCompletionPercentage, 30);
    
    // Calculate dependency statistics
    const completedTaskIds = new Set(data.tasks.filter(t => 
      t.status === 'done' || t.status === 'completed').map(t => t.id));
      
    const tasksWithNoDeps = data.tasks.filter(t => 
      t.status !== 'done' && 
      t.status !== 'completed' && 
      (!t.dependencies || t.dependencies.length === 0)).length;
      
    const tasksWithAllDepsSatisfied = data.tasks.filter(t => 
      t.status !== 'done' && 
      t.status !== 'completed' && 
      t.dependencies && 
      t.dependencies.length > 0 && 
      t.dependencies.every(depId => completedTaskIds.has(depId))).length;
      
    const tasksWithUnsatisfiedDeps = data.tasks.filter(t => 
      t.status !== 'done' && 
      t.status !== 'completed' && 
      t.dependencies && 
      t.dependencies.length > 0 && 
      !t.dependencies.every(depId => completedTaskIds.has(depId))).length;
    
    // Calculate total tasks ready to work on (no deps + satisfied deps)
    const tasksReadyToWork = tasksWithNoDeps + tasksWithAllDepsSatisfied;
    
    // Calculate most depended-on tasks
    const dependencyCount = {};
    data.tasks.forEach(task => {
      if (task.dependencies && task.dependencies.length > 0) {
        task.dependencies.forEach(depId => {
          dependencyCount[depId] = (dependencyCount[depId] || 0) + 1;
        });
      }
    });
    
    // Find the most depended-on task
    let mostDependedOnTaskId = null;
    let maxDependents = 0;
    
    for (const [taskId, count] of Object.entries(dependencyCount)) {
      if (count > maxDependents) {
        maxDependents = count;
        mostDependedOnTaskId = parseInt(taskId);
      }
    }
    
    // Get the most depended-on task
    const mostDependedOnTask = mostDependedOnTaskId !== null 
      ? data.tasks.find(t => t.id === mostDependedOnTaskId) 
      : null;
    
    // Calculate average dependencies per task
    const totalDependencies = data.tasks.reduce((sum, task) => 
      sum + (task.dependencies ? task.dependencies.length : 0), 0);
    const avgDependenciesPerTask = totalDependencies / data.tasks.length;
    
    // Find next task to work on
    const nextTask = findNextTask(data.tasks);
    const nextTaskInfo = nextTask ? 
      `ID: ${chalk.cyan(nextTask.id)} - ${chalk.white.bold(truncate(nextTask.title, 40))}\n` +
      `Priority: ${chalk.white(nextTask.priority || 'medium')}  Dependencies: ${formatDependenciesWithStatus(nextTask.dependencies, data.tasks, true)}` : 
      chalk.yellow('No eligible tasks found. All tasks are either completed or have unsatisfied dependencies.');
    
    // Get terminal width - more reliable method
    let terminalWidth;
    try {
      // Try to get the actual terminal columns
      terminalWidth = process.stdout.columns;
    } catch (e) {
      // Fallback if columns cannot be determined
      log('debug', 'Could not determine terminal width, using default');
    }
    // Ensure we have a reasonable default if detection fails
    terminalWidth = terminalWidth || 80;
    
    // Ensure terminal width is at least a minimum value to prevent layout issues
    terminalWidth = Math.max(terminalWidth, 80);
    
    // Create dashboard content
    const projectDashboardContent = 
      chalk.white.bold('Project Dashboard') + '\n' +
      `Tasks Progress: ${chalk.greenBright(taskProgressBar)} ${completionPercentage.toFixed(0)}%\n` +
      `Done: ${chalk.green(doneCount)}  In Progress: ${chalk.blue(inProgressCount)}  Pending: ${chalk.yellow(pendingCount)}  Blocked: ${chalk.red(blockedCount)}  Deferred: ${chalk.gray(deferredCount)}\n\n` +
      `Subtasks Progress: ${chalk.cyan(subtaskProgressBar)} ${subtaskCompletionPercentage.toFixed(0)}%\n` +
      `Completed: ${chalk.green(completedSubtasks)}/${totalSubtasks}  Remaining: ${chalk.yellow(totalSubtasks - completedSubtasks)}\n\n` +
      chalk.cyan.bold('Priority Breakdown:') + '\n' +
      `${chalk.red('•')} ${chalk.white('High priority:')} ${data.tasks.filter(t => t.priority === 'high').length}\n` +
      `${chalk.yellow('•')} ${chalk.white('Medium priority:')} ${data.tasks.filter(t => t.priority === 'medium').length}\n` +
      `${chalk.green('•')} ${chalk.white('Low priority:')} ${data.tasks.filter(t => t.priority === 'low').length}`;
    
    const dependencyDashboardContent =  
      chalk.white.bold('Dependency Status & Next Task') + '\n' +
      chalk.cyan.bold('Dependency Metrics:') + '\n' +
      `${chalk.green('•')} ${chalk.white('Tasks with no dependencies:')} ${tasksWithNoDeps}\n` +
      `${chalk.green('•')} ${chalk.white('Tasks ready to work on:')} ${tasksReadyToWork}\n` +
      `${chalk.yellow('•')} ${chalk.white('Tasks blocked by dependencies:')} ${tasksWithUnsatisfiedDeps}\n` +
      `${chalk.magenta('•')} ${chalk.white('Most depended-on task:')} ${mostDependedOnTask ? chalk.cyan(`#${mostDependedOnTaskId} (${maxDependents} dependents)`) : chalk.gray('None')}\n` +
      `${chalk.blue('•')} ${chalk.white('Avg dependencies per task:')} ${avgDependenciesPerTask.toFixed(1)}\n\n` +
      chalk.cyan.bold('Next Task to Work On:') + '\n' +
      `ID: ${chalk.cyan(nextTask ? nextTask.id : 'N/A')} - ${nextTask ? chalk.white.bold(truncate(nextTask.title, 40)) : chalk.yellow('No task available')}\n` +
      `Priority: ${nextTask ? chalk.white(nextTask.priority || 'medium') : ''}  Dependencies: ${nextTask ? formatDependenciesWithStatus(nextTask.dependencies, data.tasks, true) : ''}`;
    
    // Calculate width for side-by-side display
    // Box borders, padding take approximately 4 chars on each side
    const minDashboardWidth = 50; // Minimum width for dashboard
    const minDependencyWidth = 50; // Minimum width for dependency dashboard
    const totalMinWidth = minDashboardWidth + minDependencyWidth + 4; // Extra 4 chars for spacing
    
    // If terminal is wide enough, show boxes side by side with responsive widths
    if (terminalWidth >= totalMinWidth) {
      // Calculate widths proportionally for each box - use exact 50% width each
      const availableWidth = terminalWidth;
      const halfWidth = Math.floor(availableWidth / 2);
      
      // Account for border characters (2 chars on each side)
      const boxContentWidth = halfWidth - 4;
      
      // Create boxen options with precise widths
      const dashboardBox = boxen(
        projectDashboardContent,
        { 
          padding: 1, 
          borderColor: 'blue', 
          borderStyle: 'round', 
          width: boxContentWidth,
          dimBorder: false
        }
      );
      
      const dependencyBox = boxen(
        dependencyDashboardContent,
        { 
          padding: 1, 
          borderColor: 'magenta', 
          borderStyle: 'round', 
          width: boxContentWidth,
          dimBorder: false
        }
      );
      
      // Create a better side-by-side layout with exact spacing
      const dashboardLines = dashboardBox.split('\n');
      const dependencyLines = dependencyBox.split('\n');
      
      // Make sure both boxes have the same height
      const maxHeight = Math.max(dashboardLines.length, dependencyLines.length);
      
      // For each line of output, pad the dashboard line to exactly halfWidth chars
      // This ensures the dependency box starts at exactly the right position
      const combinedLines = [];
      for (let i = 0; i < maxHeight; i++) {
        // Get the dashboard line (or empty string if we've run out of lines)
        const dashLine = i < dashboardLines.length ? dashboardLines[i] : '';
        // Get the dependency line (or empty string if we've run out of lines)
        const depLine = i < dependencyLines.length ? dependencyLines[i] : '';
        
        // Remove any trailing spaces from dashLine before padding to exact width
        const trimmedDashLine = dashLine.trimEnd();
        // Pad the dashboard line to exactly halfWidth chars with no extra spaces
        const paddedDashLine = trimmedDashLine.padEnd(halfWidth, ' ');
        
        // Join the lines with no space in between
        combinedLines.push(paddedDashLine + depLine);
      }
      
      // Join all lines and output
      console.log(combinedLines.join('\n'));
    } else {
      // Terminal too narrow, show boxes stacked vertically
      const dashboardBox = boxen(
        projectDashboardContent,
        { padding: 1, borderColor: 'blue', borderStyle: 'round', margin: { top: 0, bottom: 1 } }
      );
      
      const dependencyBox = boxen(
        dependencyDashboardContent,
        { padding: 1, borderColor: 'magenta', borderStyle: 'round', margin: { top: 0, bottom: 1 } }
      );
      
      // Display stacked vertically
      console.log(dashboardBox);
      console.log(dependencyBox);
    }
    
    if (filteredTasks.length === 0) {
      console.log(boxen(
        statusFilter 
          ? chalk.yellow(`No tasks with status '${statusFilter}' found`) 
          : chalk.yellow('No tasks found'),
        { padding: 1, borderColor: 'yellow', borderStyle: 'round' }
      ));
      return;
    }
    
    // COMPLETELY REVISED TABLE APPROACH
    // Define percentage-based column widths and calculate actual widths
    // Adjust percentages based on content type and user requirements

    // Adjust ID width if showing subtasks (subtask IDs are longer: e.g., "1.2")
    const idWidthPct = withSubtasks ? 10 : 7;
    
    // Calculate max status length to accommodate "in-progress"
    const statusWidthPct = 15;
    
    // Increase priority column width as requested
    const priorityWidthPct = 12;
    
    // Make dependencies column smaller as requested (-20%)
    const depsWidthPct = 20;
    
    // Calculate title/description width as remaining space (+20% from dependencies reduction)
    const titleWidthPct = 100 - idWidthPct - statusWidthPct - priorityWidthPct - depsWidthPct;
    
    // Allow 10 characters for borders and padding
    const availableWidth = terminalWidth - 10;
    
    // Calculate actual column widths based on percentages
    const idWidth = Math.floor(availableWidth * (idWidthPct / 100));
    const statusWidth = Math.floor(availableWidth * (statusWidthPct / 100));
    const priorityWidth = Math.floor(availableWidth * (priorityWidthPct / 100));
    const depsWidth = Math.floor(availableWidth * (depsWidthPct / 100));
    const titleWidth = Math.floor(availableWidth * (titleWidthPct / 100));
    
    // Create a table with correct borders and spacing
    const table = new Table({
      head: [
        chalk.cyan.bold('ID'),
        chalk.cyan.bold('Title'),
        chalk.cyan.bold('Status'),
        chalk.cyan.bold('Priority'),
        chalk.cyan.bold('Dependencies')
      ],
      colWidths: [idWidth, titleWidth, statusWidth, priorityWidth, depsWidth],
      style: {
        head: [], // No special styling for header
        border: [], // No special styling for border
        compact: false // Use default spacing
      },
      wordWrap: true,
      wrapOnWordBoundary: true,
    });
    
    // Process tasks for the table
    filteredTasks.forEach(task => {
      // Format dependencies with status indicators (colored)
      let depText = 'None';
      if (task.dependencies && task.dependencies.length > 0) {
        // Use the proper formatDependenciesWithStatus function for colored status
        depText = formatDependenciesWithStatus(task.dependencies, data.tasks, true);
      } else {
        depText = chalk.gray('None');
      }
      
      // Clean up any ANSI codes or confusing characters
      const cleanTitle = task.title.replace(/\n/g, ' ');
      
      // Get priority color
      const priorityColor = {
        'high': chalk.red,
        'medium': chalk.yellow,
        'low': chalk.gray
      }[task.priority || 'medium'] || chalk.white;
      
      // Format status
      const status = getStatusWithColor(task.status, true);
      
      // Add the row without truncating dependencies
      table.push([
        task.id.toString(),
        truncate(cleanTitle, titleWidth - 3),
        status,
        priorityColor(truncate(task.priority || 'medium', priorityWidth - 2)),
        depText // No truncation for dependencies
      ]);
      
      // Add subtasks if requested
      if (withSubtasks && task.subtasks && task.subtasks.length > 0) {
        task.subtasks.forEach(subtask => {
          // Format subtask dependencies with status indicators
          let subtaskDepText = 'None';
          if (subtask.dependencies && subtask.dependencies.length > 0) {
            // Handle both subtask-to-subtask and subtask-to-task dependencies
            const formattedDeps = subtask.dependencies.map(depId => {
              // Check if it's a dependency on another subtask
              if (typeof depId === 'number' && depId < 100) {
                const foundSubtask = task.subtasks.find(st => st.id === depId);
                if (foundSubtask) {
                  const isDone = foundSubtask.status === 'done' || foundSubtask.status === 'completed';
                  const isInProgress = foundSubtask.status === 'in-progress';
                  
                  // Use consistent color formatting instead of emojis
                  if (isDone) {
                    return chalk.green.bold(`${task.id}.${depId}`);
                  } else if (isInProgress) {
                    return chalk.hex('#FFA500').bold(`${task.id}.${depId}`);
                  } else {
                    return chalk.red.bold(`${task.id}.${depId}`);
                  }
                }
              }
              // Default to regular task dependency
              const depTask = data.tasks.find(t => t.id === depId);
              if (depTask) {
                const isDone = depTask.status === 'done' || depTask.status === 'completed';
                const isInProgress = depTask.status === 'in-progress';
                // Use the same color scheme as in formatDependenciesWithStatus
                if (isDone) {
                  return chalk.green.bold(`${depId}`);
                } else if (isInProgress) {
                  return chalk.hex('#FFA500').bold(`${depId}`);
                } else {
                  return chalk.red.bold(`${depId}`);
                }
              }
              return chalk.cyan(depId.toString());
            }).join(', ');
            
            subtaskDepText = formattedDeps || chalk.gray('None');
          }
          
          // Add the subtask row without truncating dependencies
          table.push([
            `${task.id}.${subtask.id}`,
            chalk.dim(`└─ ${truncate(subtask.title, titleWidth - 5)}`),
            getStatusWithColor(subtask.status, true),
            chalk.dim('-'),
            subtaskDepText // No truncation for dependencies
          ]);
        });
      }
    });
    
    // Ensure we output the table even if it had to wrap
    try {
      console.log(table.toString());
    } catch (err) {
      log('error', `Error rendering table: ${err.message}`);
      
      // Fall back to simpler output
      console.log(chalk.yellow('\nFalling back to simple task list due to terminal width constraints:'));
      filteredTasks.forEach(task => {
        console.log(`${chalk.cyan(task.id)}: ${chalk.white(task.title)} - ${getStatusWithColor(task.status)}`);
      });
    }
    
    // Show filter info if applied
    if (statusFilter) {
      console.log(chalk.yellow(`\nFiltered by status: ${statusFilter}`));
      console.log(chalk.yellow(`Showing ${filteredTasks.length} of ${totalTasks} tasks`));
    }
    
    // Define priority colors
    const priorityColors = {
      'high': chalk.red.bold,
        'medium': chalk.yellow,
        'low': chalk.gray
    };
    
    // Show next task box in a prominent color
    if (nextTask) {
      // Prepare subtasks section if they exist
      let subtasksSection = '';
      if (nextTask.subtasks && nextTask.subtasks.length > 0) {
        subtasksSection = `\n\n${chalk.white.bold('Subtasks:')}\n`;
        subtasksSection += nextTask.subtasks.map(subtask => {
          // Using a more simplified format for subtask status display
          const status = subtask.status || 'pending';
          const statusColors = {
            'done': chalk.green,
            'completed': chalk.green,
            'pending': chalk.yellow,
            'in-progress': chalk.blue,
            'deferred': chalk.gray,
            'blocked': chalk.red
          };
          const statusColor = statusColors[status.toLowerCase()] || chalk.white;
          return `${chalk.cyan(`${nextTask.id}.${subtask.id}`)} [${statusColor(status)}] ${subtask.title}`;
        }).join('\n');
      }
      
      console.log(boxen(
        chalk.hex('#FF8800').bold(`🔥 Next Task to Work On: #${nextTask.id} - ${nextTask.title}`) + '\n\n' +
        `${chalk.white('Priority:')} ${priorityColors[nextTask.priority || 'medium'](nextTask.priority || 'medium')}   ${chalk.white('Status:')} ${getStatusWithColor(nextTask.status, true)}\n` +
        `${chalk.white('Dependencies:')} ${nextTask.dependencies && nextTask.dependencies.length > 0 ? formatDependenciesWithStatus(nextTask.dependencies, data.tasks, true) : chalk.gray('None')}\n\n` +
        `${chalk.white('Description:')} ${nextTask.description}` +
        subtasksSection + '\n\n' +
        `${chalk.cyan('Start working:')} ${chalk.yellow(`task-master set-status --id=${nextTask.id} --status=in-progress`)}\n` +
        `${chalk.cyan('View details:')} ${chalk.yellow(`task-master show ${nextTask.id}`)}`,
        { 
          padding: { left: 2, right: 2, top: 1, bottom: 1 },
          borderColor: '#FF8800', 
          borderStyle: 'round', 
          margin: { top: 1, bottom: 1 },
          title: '⚡ RECOMMENDED NEXT TASK ⚡',
          titleAlignment: 'center',
          width: terminalWidth - 4, // Use full terminal width minus a small margin
          fullscreen: false // Keep it expandable but not literally fullscreen
        }
      ));
    } else {
      console.log(boxen(
        chalk.hex('#FF8800').bold('No eligible next task found') + '\n\n' +
        'All pending tasks have dependencies that are not yet completed, or all tasks are done.',
        { 
          padding: 1, 
          borderColor: '#FF8800', 
          borderStyle: 'round', 
          margin: { top: 1, bottom: 1 },
          title: '⚡ NEXT TASK ⚡',
          titleAlignment: 'center',
          width: terminalWidth - 4, // Use full terminal width minus a small margin
        }
      ));
    }
    
    // Show next steps
    console.log(boxen(
      chalk.white.bold('Suggested Next Steps:') + '\n\n' +
      `${chalk.cyan('1.')} Run ${chalk.yellow('task-master next')} to see what to work on next\n` +
      `${chalk.cyan('2.')} Run ${chalk.yellow('task-master expand --id=<id>')} to break down a task into subtasks\n` +
      `${chalk.cyan('3.')} Run ${chalk.yellow('task-master set-status --id=<id> --status=done')} to mark a task as complete`,
      { padding: 1, borderColor: 'gray', borderStyle: 'round', margin: { top: 1 } }
    ));
  } catch (error) {
    log('error', `Error listing tasks: ${error.message}`);
    console.error(chalk.red(`Error: ${error.message}`));
    
    if (CONFIG.debug) {
      console.error(error);
    }
    
    process.exit(1);
  }
}

/**
 * Safely apply chalk coloring, stripping ANSI codes when calculating string length
 * @param {string} text - Original text
 * @param {Function} colorFn - Chalk color function
 * @param {number} maxLength - Maximum allowed length
 * @returns {string} Colored text that won't break table layout
 */
function safeColor(text, colorFn, maxLength = 0) {
  if (!text) return '';
  
  // If maxLength is provided, truncate the text first
  const baseText = maxLength > 0 ? truncate(text, maxLength) : text;
  
  // Apply color function if provided, otherwise return as is
  return colorFn ? colorFn(baseText) : baseText;
}

/**
 * Expand a task with subtasks
 * @param {number} taskId - Task ID to expand
 * @param {number} numSubtasks - Number of subtasks to generate
 * @param {boolean} useResearch - Whether to use research (Perplexity)
 * @param {string} additionalContext - Additional context
 */
async function expandTask(taskId, numSubtasks = CONFIG.defaultSubtasks, useResearch = false, additionalContext = '') {
  try {
    displayBanner();
    
    // Load tasks
    const tasksPath = path.join(process.cwd(), 'tasks', 'tasks.json');
    log('info', `Loading tasks from ${tasksPath}...`);
    
    const data = readJSON(tasksPath);
    if (!data || !data.tasks) {
      throw new Error(`No valid tasks found in ${tasksPath}`);
    }
    
    // Find the task
    const task = data.tasks.find(t => t.id === taskId);
    if (!task) {
      throw new Error(`Task ${taskId} not found`);
    }
    
    // Check if the task is already completed
    if (task.status === 'done' || task.status === 'completed') {
      log('warn', `Task ${taskId} is already marked as "${task.status}". Skipping expansion.`);
      console.log(chalk.yellow(`Task ${taskId} is already marked as "${task.status}". Skipping expansion.`));
      return;
    }
    
    // Check for complexity report
    log('info', 'Checking for complexity analysis...');
    const complexityReport = readComplexityReport();
    let taskAnalysis = null;
    
    if (complexityReport) {
      taskAnalysis = findTaskInComplexityReport(complexityReport, taskId);
      
      if (taskAnalysis) {
        log('info', `Found complexity analysis for task ${taskId}: Score ${taskAnalysis.complexityScore}/10`);
        
        // Use recommended number of subtasks if available and not overridden
        if (taskAnalysis.recommendedSubtasks && numSubtasks === CONFIG.defaultSubtasks) {
          numSubtasks = taskAnalysis.recommendedSubtasks;
          log('info', `Using recommended number of subtasks: ${numSubtasks}`);
        }
        
        // Use expansion prompt from analysis as additional context if available
        if (taskAnalysis.expansionPrompt && !additionalContext) {
          additionalContext = taskAnalysis.expansionPrompt;
          log('info', 'Using expansion prompt from complexity analysis');
        }
      } else {
        log('info', `No complexity analysis found for task ${taskId}`);
      }
    }
    
    console.log(boxen(
      chalk.white.bold(`Expanding Task: #${taskId} - ${task.title}`),
      { padding: 1, borderColor: 'blue', borderStyle: 'round', margin: { top: 0, bottom: 1 } }
    ));
    
    // Check if the task already has subtasks
    if (task.subtasks && task.subtasks.length > 0) {
      log('warn', `Task ${taskId} already has ${task.subtasks.length} subtasks. Appending new subtasks.`);
      console.log(chalk.yellow(`Task ${taskId} already has ${task.subtasks.length} subtasks. New subtasks will be appended.`));
    }
    
    // Initialize subtasks array if it doesn't exist
    if (!task.subtasks) {
      task.subtasks = [];
    }
    
    // Determine the next subtask ID
    const nextSubtaskId = task.subtasks.length > 0 ? 
      Math.max(...task.subtasks.map(st => st.id)) + 1 : 1;
    
    // Generate subtasks
    let subtasks;
    if (useResearch) {
      log('info', 'Using Perplexity AI for research-backed subtask generation');
      subtasks = await generateSubtasksWithPerplexity(task, numSubtasks, nextSubtaskId, additionalContext);
    } else {
      log('info', 'Generating subtasks with Claude only');
      subtasks = await generateSubtasks(task, numSubtasks, nextSubtaskId, additionalContext);
    }
    
    // Add the subtasks to the task
    task.subtasks = [...task.subtasks, ...subtasks];
    
    // Write the updated tasks to the file
    writeJSON(tasksPath, data);
    
    // Generate individual task files
    await generateTaskFiles(tasksPath, path.dirname(tasksPath));
    
    // Display success message
    console.log(boxen(
      chalk.green(`Successfully added ${subtasks.length} subtasks to task ${taskId}`),
      { padding: 1, borderColor: 'green', borderStyle: 'round' }
    ));
    
    // Show the subtasks table
    const table = new Table({
      head: [
        chalk.cyan.bold('ID'),
        chalk.cyan.bold('Title'),
        chalk.cyan.bold('Dependencies'),
        chalk.cyan.bold('Status')
      ],
      colWidths: [8, 50, 15, 15]
    });
    
    subtasks.forEach(subtask => {
      const deps = subtask.dependencies && subtask.dependencies.length > 0 ? 
        subtask.dependencies.map(d => `${taskId}.${d}`).join(', ') : 
        chalk.gray('None');
      
      table.push([
        `${taskId}.${subtask.id}`,
        truncate(subtask.title, 47),
        deps,
        getStatusWithColor(subtask.status, true)
      ]);
    });
    
    console.log(table.toString());
    
    // Show next steps
    console.log(boxen(
      chalk.white.bold('Next Steps:') + '\n\n' +
      `${chalk.cyan('1.')} Run ${chalk.yellow(`task-master show ${taskId}`)} to see the full task with subtasks\n` +
      `${chalk.cyan('2.')} Start working on subtask: ${chalk.yellow(`task-master set-status --id=${taskId}.1 --status=in-progress`)}\n` +
      `${chalk.cyan('3.')} Mark subtask as done: ${chalk.yellow(`task-master set-status --id=${taskId}.1 --status=done`)}`,
      { padding: 1, borderColor: 'cyan', borderStyle: 'round', margin: { top: 1 } }
    ));
  } catch (error) {
    log('error', `Error expanding task: ${error.message}`);
    console.error(chalk.red(`Error: ${error.message}`));
    
    if (CONFIG.debug) {
      console.error(error);
    }
    
    process.exit(1);
  }
}

/**
 * Expand all pending tasks with subtasks
 * @param {number} numSubtasks - Number of subtasks per task
 * @param {boolean} useResearch - Whether to use research (Perplexity)
 * @param {string} additionalContext - Additional context
 * @param {boolean} forceFlag - Force regeneration for tasks with subtasks
 */
async function expandAllTasks(numSubtasks = CONFIG.defaultSubtasks, useResearch = false, additionalContext = '', forceFlag = false) {
  try {
    displayBanner();
    
    // Load tasks
    const tasksPath = path.join(process.cwd(), 'tasks', 'tasks.json');
    log('info', `Loading tasks from ${tasksPath}...`);
    
    const data = readJSON(tasksPath);
    if (!data || !data.tasks) {
      throw new Error(`No valid tasks found in ${tasksPath}`);
    }
    
    // Get complexity report if it exists
    log('info', 'Checking for complexity analysis...');
    const complexityReport = readComplexityReport();
    
    // Filter tasks that are not done and don't have subtasks (unless forced)
    const pendingTasks = data.tasks.filter(task => 
      task.status !== 'done' && 
      task.status !== 'completed' && 
      (forceFlag || !task.subtasks || task.subtasks.length === 0)
    );
    
    if (pendingTasks.length === 0) {
      log('info', 'No pending tasks found to expand');
      console.log(boxen(
        chalk.yellow('No pending tasks found to expand'),
        { padding: 1, borderColor: 'yellow', borderStyle: 'round' }
      ));
      return;
    }
    
    // Sort tasks by complexity if report exists, otherwise by ID
    let tasksToExpand = [...pendingTasks];
    
    if (complexityReport && complexityReport.complexityAnalysis) {
      log('info', 'Sorting tasks by complexity...');
      
      // Create a map of task IDs to complexity scores
      const complexityMap = new Map();
      complexityReport.complexityAnalysis.forEach(analysis => {
        complexityMap.set(analysis.taskId, analysis.complexityScore);
      });
      
      // Sort tasks by complexity score (high to low)
      tasksToExpand.sort((a, b) => {
        const scoreA = complexityMap.get(a.id) || 0;
        const scoreB = complexityMap.get(b.id) || 0;
        return scoreB - scoreA;
      });
    } else {
      // Sort by ID if no complexity report
      tasksToExpand.sort((a, b) => a.id - b.id);
    }
    
    console.log(boxen(
      chalk.white.bold(`Expanding ${tasksToExpand.length} Pending Tasks`),
      { padding: 1, borderColor: 'blue', borderStyle: 'round', margin: { top: 0, bottom: 1 } }
    ));
    
    // Show tasks to be expanded
    const table = new Table({
      head: [
        chalk.cyan.bold('ID'),
        chalk.cyan.bold('Title'),
        chalk.cyan.bold('Status'),
        chalk.cyan.bold('Complexity')
      ],
      colWidths: [5, 50, 15, 15]
    });
    
    tasksToExpand.forEach(task => {
      const taskAnalysis = complexityReport ? 
        findTaskInComplexityReport(complexityReport, task.id) : null;
      
      const complexity = taskAnalysis ? 
        getComplexityWithColor(taskAnalysis.complexityScore) + '/10' : 
        chalk.gray('Unknown');
      
      table.push([
        task.id,
        truncate(task.title, 47),
        getStatusWithColor(task.status),
        complexity
      ]);
    });
    
    console.log(table.toString());
    
    // Confirm expansion
    console.log(chalk.yellow(`\nThis will expand ${tasksToExpand.length} tasks with ${numSubtasks} subtasks each.`));
    console.log(chalk.yellow(`Research-backed generation: ${useResearch ? 'Yes' : 'No'}`));
    console.log(chalk.yellow(`Force regeneration: ${forceFlag ? 'Yes' : 'No'}`));
    
    // Expand each task
    let expandedCount = 0;
    for (const task of tasksToExpand) {
      try {
        log('info', `Expanding task ${task.id}: ${task.title}`);
        
        // Get task-specific parameters from complexity report
        let taskSubtasks = numSubtasks;
        let taskContext = additionalContext;
        
        if (complexityReport) {
          const taskAnalysis = findTaskInComplexityReport(complexityReport, task.id);
          if (taskAnalysis) {
            // Use recommended subtasks if default wasn't overridden
            if (taskAnalysis.recommendedSubtasks && numSubtasks === CONFIG.defaultSubtasks) {
              taskSubtasks = taskAnalysis.recommendedSubtasks;
              log('info', `Using recommended subtasks for task ${task.id}: ${taskSubtasks}`);
            }
            
            // Add expansion prompt if no user context was provided
            if (taskAnalysis.expansionPrompt && !additionalContext) {
              taskContext = taskAnalysis.expansionPrompt;
              log('info', `Using complexity analysis prompt for task ${task.id}`);
            }
          }
        }
        
        // Check if the task already has subtasks
        if (task.subtasks && task.subtasks.length > 0) {
          if (forceFlag) {
            log('info', `Task ${task.id} already has ${task.subtasks.length} subtasks. Clearing them due to --force flag.`);
            task.subtasks = []; // Clear existing subtasks
          } else {
            log('warn', `Task ${task.id} already has subtasks. Skipping (use --force to regenerate).`);
            continue;
          }
        }
        
        // Initialize subtasks array if it doesn't exist
        if (!task.subtasks) {
          task.subtasks = [];
        }
        
        // Determine the next subtask ID
        const nextSubtaskId = task.subtasks.length > 0 ? 
          Math.max(...task.subtasks.map(st => st.id)) + 1 : 1;
        
        // Generate subtasks
        let subtasks;
        if (useResearch) {
          subtasks = await generateSubtasksWithPerplexity(task, taskSubtasks, nextSubtaskId, taskContext);
        } else {
          subtasks = await generateSubtasks(task, taskSubtasks, nextSubtaskId, taskContext);
        }
        
        // Add the subtasks to the task
        task.subtasks = [...task.subtasks, ...subtasks];
        expandedCount++;
      } catch (error) {
        log('error', `Error expanding task ${task.id}: ${error.message}`);
        console.error(chalk.red(`Error expanding task ${task.id}: ${error.message}`));
        continue;
      }
    }
    
    // Write the updated tasks to the file
    writeJSON(tasksPath, data);
    
    // Generate individual task files
    await generateTaskFiles(tasksPath, path.dirname(tasksPath));
    
    // Display success message
    console.log(boxen(
      chalk.green(`Successfully expanded ${expandedCount} of ${tasksToExpand.length} tasks`),
      { padding: 1, borderColor: 'green', borderStyle: 'round' }
    ));
    
    // Show next steps
    console.log(boxen(
      chalk.white.bold('Next Steps:') + '\n\n' +
      `${chalk.cyan('1.')} Run ${chalk.yellow('task-master list --with-subtasks')} to see all tasks with subtasks\n` +
      `${chalk.cyan('2.')} Run ${chalk.yellow('task-master next')} to see what to work on next`,
      { padding: 1, borderColor: 'cyan', borderStyle: 'round', margin: { top: 1 } }
    ));
  } catch (error) {
    log('error', `Error expanding tasks: ${error.message}`);
    console.error(chalk.red(`Error: ${error.message}`));
    
    if (CONFIG.debug) {
      console.error(error);
    }
    
    process.exit(1);
  }
}

/**
 * Clear subtasks from specified tasks
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {string} taskIds - Task IDs to clear subtasks from
 */
function clearSubtasks(tasksPath, taskIds) {
  displayBanner();
  
  log('info', `Reading tasks from ${tasksPath}...`);
  const data = readJSON(tasksPath);
  if (!data || !data.tasks) {
    log('error', "No valid tasks found.");
    process.exit(1);
  }

  console.log(boxen(
    chalk.white.bold('Clearing Subtasks'), 
    { padding: 1, borderColor: 'blue', borderStyle: 'round', margin: { top: 1, bottom: 1 } }
  ));

  // Handle multiple task IDs (comma-separated)
  const taskIdArray = taskIds.split(',').map(id => id.trim());
  let clearedCount = 0;
  
  // Create a summary table for the cleared subtasks
  const summaryTable = new Table({
    head: [
      chalk.cyan.bold('Task ID'), 
      chalk.cyan.bold('Task Title'), 
      chalk.cyan.bold('Subtasks Cleared')
    ],
    colWidths: [10, 50, 20],
    style: { head: [], border: [] }
  });

  taskIdArray.forEach(taskId => {
    const id = parseInt(taskId, 10);
    if (isNaN(id)) {
      log('error', `Invalid task ID: ${taskId}`);
      return;
    }

    const task = data.tasks.find(t => t.id === id);
    if (!task) {
      log('error', `Task ${id} not found`);
      return;
    }

    if (!task.subtasks || task.subtasks.length === 0) {
      log('info', `Task ${id} has no subtasks to clear`);
      summaryTable.push([
        id.toString(),
        truncate(task.title, 47),
        chalk.yellow('No subtasks')
      ]);
      return;
    }

    const subtaskCount = task.subtasks.length;
    task.subtasks = [];
    clearedCount++;
    log('info', `Cleared ${subtaskCount} subtasks from task ${id}`);
    
    summaryTable.push([
      id.toString(),
      truncate(task.title, 47),
      chalk.green(`${subtaskCount} subtasks cleared`)
    ]);
  });

  if (clearedCount > 0) {
    writeJSON(tasksPath, data);
    
    // Show summary table
    console.log(boxen(
      chalk.white.bold('Subtask Clearing Summary:'),
      { padding: { left: 2, right: 2, top: 0, bottom: 0 }, margin: { top: 1, bottom: 0 }, borderColor: 'blue', borderStyle: 'round' }
    ));
    console.log(summaryTable.toString());
    
    // Regenerate task files to reflect changes
    log('info', "Regenerating task files...");
    generateTaskFiles(tasksPath, path.dirname(tasksPath));
    
    // Success message
    console.log(boxen(
      chalk.green(`Successfully cleared subtasks from ${chalk.bold(clearedCount)} task(s)`),
      { padding: 1, borderColor: 'green', borderStyle: 'round', margin: { top: 1 } }
    ));
    
    // Next steps suggestion
    console.log(boxen(
      chalk.white.bold('Next Steps:') + '\n\n' +
      `${chalk.cyan('1.')} Run ${chalk.yellow('task-master expand --id=<id>')} to generate new subtasks\n` +
      `${chalk.cyan('2.')} Run ${chalk.yellow('task-master list --with-subtasks')} to verify changes`,
      { padding: 1, borderColor: 'cyan', borderStyle: 'round', margin: { top: 1 } }
    ));
    
  } else {
    console.log(boxen(
      chalk.yellow('No subtasks were cleared'),
      { padding: 1, borderColor: 'yellow', borderStyle: 'round', margin: { top: 1 } }
    ));
  }
}

/**
 * Add a new task using AI
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {string} prompt - Description of the task to add
 * @param {Array} dependencies - Task dependencies
 * @param {string} priority - Task priority
 * @returns {number} The new task ID
 */
async function addTask(tasksPath, prompt, dependencies = [], priority = 'medium') {
  displayBanner();
  
  // Read the existing tasks
  const data = readJSON(tasksPath);
  if (!data || !data.tasks) {
    log('error', "Invalid or missing tasks.json.");
    process.exit(1);
  }
  
  // Find the highest task ID to determine the next ID
  const highestId = Math.max(...data.tasks.map(t => t.id));
  const newTaskId = highestId + 1;
  
  console.log(boxen(
    chalk.white.bold(`Creating New Task #${newTaskId}`),
    { padding: 1, borderColor: 'blue', borderStyle: 'round', margin: { top: 1, bottom: 1 } }
  ));
  
  // Validate dependencies before proceeding
  const invalidDeps = dependencies.filter(depId => {
    return !data.tasks.some(t => t.id === depId);
  });
  
  if (invalidDeps.length > 0) {
    log('warn', `The following dependencies do not exist: ${invalidDeps.join(', ')}`);
    log('info', 'Removing invalid dependencies...');
    dependencies = dependencies.filter(depId => !invalidDeps.includes(depId));
  }
  
  // Create the system prompt for Claude
  const systemPrompt = "You are a helpful assistant that creates well-structured tasks for a software development project. Generate a single new task based on the user's description.";
  
  // Create the user prompt with context from existing tasks
  let contextTasks = '';
  if (dependencies.length > 0) {
    // Provide context for the dependent tasks
    const dependentTasks = data.tasks.filter(t => dependencies.includes(t.id));
    contextTasks = `\nThis task depends on the following tasks:\n${dependentTasks.map(t => 
      `- Task ${t.id}: ${t.title} - ${t.description}`).join('\n')}`;
  } else {
    // Provide a few recent tasks as context
    const recentTasks = [...data.tasks].sort((a, b) => b.id - a.id).slice(0, 3);
    contextTasks = `\nRecent tasks in the project:\n${recentTasks.map(t => 
      `- Task ${t.id}: ${t.title} - ${t.description}`).join('\n')}`;
  }
  
  const taskStructure = `
  {
    "title": "Task title goes here",
    "description": "A concise one or two sentence description of what the task involves",
    "details": "In-depth details including specifics on implementation, considerations, and anything important for the developer to know. This should be detailed enough to guide implementation.",
    "testStrategy": "A detailed approach for verifying the task has been correctly implemented. Include specific test cases or validation methods."
  }`;
  
  const userPrompt = `Create a comprehensive new task (Task #${newTaskId}) for a software development project based on this description: "${prompt}"
  
  ${contextTasks}
  
  Return your answer as a single JSON object with the following structure:
  ${taskStructure}
  
  Don't include the task ID, status, dependencies, or priority as those will be added automatically.
  Make sure the details and test strategy are thorough and specific.
  
  IMPORTANT: Return ONLY the JSON object, nothing else.`;
  
  // Start the loading indicator
  const loadingIndicator = startLoadingIndicator('Generating new task with Claude AI...');
  
  let fullResponse = '';
  let streamingInterval = null;

  try {
    // Call Claude with streaming enabled
    const stream = await anthropic.messages.create({
      max_tokens: CONFIG.maxTokens,
      model: CONFIG.model,
      temperature: CONFIG.temperature,
      messages: [{ role: "user", content: userPrompt }],
      system: systemPrompt,
      stream: true
    });
    
    // Update loading indicator to show streaming progress
    let dotCount = 0;
    streamingInterval = setInterval(() => {
      readline.cursorTo(process.stdout, 0);
      process.stdout.write(`Receiving streaming response from Claude${'.'.repeat(dotCount)}`);
      dotCount = (dotCount + 1) % 4;
    }, 500);
    
    // Process the stream
    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta' && chunk.delta.text) {
        fullResponse += chunk.delta.text;
      }
    }
    
    if (streamingInterval) clearInterval(streamingInterval);
    stopLoadingIndicator(loadingIndicator);
    
    log('info', "Completed streaming response from Claude API!");
    log('debug', `Streaming response length: ${fullResponse.length} characters`);
    
    // Parse the response - handle potential JSON formatting issues
    let taskData;
    try {
      // Check if the response is wrapped in a code block
      const jsonMatch = fullResponse.match(/```(?:json)?([^`]+)```/);
      const jsonContent = jsonMatch ? jsonMatch[1] : fullResponse;
      
      // Parse the JSON
      taskData = JSON.parse(jsonContent);
      
      // Check that we have the required fields
      if (!taskData.title || !taskData.description) {
        throw new Error("Missing required fields in the generated task");
      }
    } catch (error) {
      log('error', "Failed to parse Claude's response as valid task JSON:", error);
      log('debug', "Response content:", fullResponse);
      process.exit(1);
    }
    
    // Create the new task object
    const newTask = {
      id: newTaskId,
      title: taskData.title,
      description: taskData.description,
      status: "pending",
      dependencies: dependencies,
      priority: priority,
      details: taskData.details || "",
      testStrategy: taskData.testStrategy || "Manually verify the implementation works as expected."
    };
    
    // Add the new task to the tasks array
    data.tasks.push(newTask);
    
    // Validate dependencies in the entire task set
    log('info', "Validating dependencies after adding new task...");
    validateAndFixDependencies(data, null);
    
    // Write the updated tasks back to the file
    writeJSON(tasksPath, data);
    
    // Show success message
    const successBox = boxen(
      chalk.green(`Successfully added new task #${newTaskId}:\n`) +
      chalk.white.bold(newTask.title) + "\n\n" +
      chalk.white(newTask.description),
      { padding: 1, borderColor: 'green', borderStyle: 'round', margin: { top: 1 } }
    );
    console.log(successBox);
    
    // Next steps suggestion
    console.log(boxen(
      chalk.white.bold('Next Steps:') + '\n\n' +
      `${chalk.cyan('1.')} Run ${chalk.yellow('task-master generate')} to update task files\n` +
      `${chalk.cyan('2.')} Run ${chalk.yellow('task-master expand --id=' + newTaskId)} to break it down into subtasks\n` +
      `${chalk.cyan('3.')} Run ${chalk.yellow('task-master list --with-subtasks')} to see all tasks`,
      { padding: 1, borderColor: 'cyan', borderStyle: 'round', margin: { top: 1 } }
    ));
    
    return newTaskId;
  } catch (error) {
    if (streamingInterval) clearInterval(streamingInterval);
    stopLoadingIndicator(loadingIndicator);
    log('error', "Error generating task:", error.message);
    process.exit(1);
  }
}

/**
 * Find the next pending task based on dependencies
 * @param {Object[]} tasks - The array of tasks
 * @returns {Object|null} The next task to work on or null if no eligible tasks
 */
function findNextTask(tasks) {
  // Get all completed task IDs
  const completedTaskIds = new Set(
    tasks
      .filter(t => t.status === 'done' || t.status === 'completed')
      .map(t => t.id)
  );
  
  // Filter for pending tasks whose dependencies are all satisfied
  const eligibleTasks = tasks.filter(task => 
    (task.status === 'pending' || task.status === 'in-progress') && 
    task.dependencies && // Make sure dependencies array exists
    task.dependencies.every(depId => completedTaskIds.has(depId))
  );
  
  if (eligibleTasks.length === 0) {
    return null;
  }
  
  // Sort eligible tasks by:
  // 1. Priority (high > medium > low)
  // 2. Dependencies count (fewer dependencies first)
  // 3. ID (lower ID first)
  const priorityValues = { 'high': 3, 'medium': 2, 'low': 1 };
  
  const nextTask = eligibleTasks.sort((a, b) => {
    // Sort by priority first
    const priorityA = priorityValues[a.priority || 'medium'] || 2;
    const priorityB = priorityValues[b.priority || 'medium'] || 2;
    
    if (priorityB !== priorityA) {
      return priorityB - priorityA; // Higher priority first
    }
    
    // If priority is the same, sort by dependency count
    if (a.dependencies && b.dependencies && a.dependencies.length !== b.dependencies.length) {
      return a.dependencies.length - b.dependencies.length; // Fewer dependencies first
    }
    
    // If dependency count is the same, sort by ID
    return a.id - b.id; // Lower ID first
  })[0]; // Return the first (highest priority) task
  
  return nextTask;
}

/**
 * Add a subtask to a parent task
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {number|string} parentId - ID of the parent task
 * @param {number|string|null} existingTaskId - ID of an existing task to convert to subtask (optional)
 * @param {Object} newSubtaskData - Data for creating a new subtask (used if existingTaskId is null)
 * @param {boolean} generateFiles - Whether to regenerate task files after adding the subtask
 * @returns {Object} The newly created or converted subtask
 */
async function addSubtask(tasksPath, parentId, existingTaskId = null, newSubtaskData = null, generateFiles = true) {
  try {
    log('info', `Adding subtask to parent task ${parentId}...`);
    
    // Read the existing tasks
    const data = readJSON(tasksPath);
    if (!data || !data.tasks) {
      throw new Error(`Invalid or missing tasks file at ${tasksPath}`);
    }
    
    // Convert parent ID to number
    const parentIdNum = parseInt(parentId, 10);
    
    // Find the parent task
    const parentTask = data.tasks.find(t => t.id === parentIdNum);
    if (!parentTask) {
      throw new Error(`Parent task with ID ${parentIdNum} not found`);
    }
    
    // Initialize subtasks array if it doesn't exist
    if (!parentTask.subtasks) {
      parentTask.subtasks = [];
    }
    
    let newSubtask;
    
    // Case 1: Convert an existing task to a subtask
    if (existingTaskId !== null) {
      const existingTaskIdNum = parseInt(existingTaskId, 10);
      
      // Find the existing task
      const existingTaskIndex = data.tasks.findIndex(t => t.id === existingTaskIdNum);
      if (existingTaskIndex === -1) {
        throw new Error(`Task with ID ${existingTaskIdNum} not found`);
      }
      
      const existingTask = data.tasks[existingTaskIndex];
      
      // Check if task is already a subtask
      if (existingTask.parentTaskId) {
        throw new Error(`Task ${existingTaskIdNum} is already a subtask of task ${existingTask.parentTaskId}`);
      }
      
      // Check for circular dependency
      if (existingTaskIdNum === parentIdNum) {
        throw new Error(`Cannot make a task a subtask of itself`);
      }
      
      // Check if parent task is a subtask of the task we're converting
      // This would create a circular dependency
      if (isTaskDependentOn(data.tasks, parentTask, existingTaskIdNum)) {
        throw new Error(`Cannot create circular dependency: task ${parentIdNum} is already a subtask or dependent of task ${existingTaskIdNum}`);
      }
      
      // Find the highest subtask ID to determine the next ID
      const highestSubtaskId = parentTask.subtasks.length > 0 
        ? Math.max(...parentTask.subtasks.map(st => st.id))
        : 0;
      const newSubtaskId = highestSubtaskId + 1;
      
      // Clone the existing task to be converted to a subtask
      newSubtask = { ...existingTask, id: newSubtaskId, parentTaskId: parentIdNum };
      
      // Add to parent's subtasks
      parentTask.subtasks.push(newSubtask);
      
      // Remove the task from the main tasks array
      data.tasks.splice(existingTaskIndex, 1);
      
      log('info', `Converted task ${existingTaskIdNum} to subtask ${parentIdNum}.${newSubtaskId}`);
    }
    // Case 2: Create a new subtask
    else if (newSubtaskData) {
      // Find the highest subtask ID to determine the next ID
      const highestSubtaskId = parentTask.subtasks.length > 0 
        ? Math.max(...parentTask.subtasks.map(st => st.id))
        : 0;
      const newSubtaskId = highestSubtaskId + 1;
      
      // Create the new subtask object
      newSubtask = {
        id: newSubtaskId,
        title: newSubtaskData.title,
        description: newSubtaskData.description || '',
        details: newSubtaskData.details || '',
        status: newSubtaskData.status || 'pending',
        dependencies: newSubtaskData.dependencies || [],
        parentTaskId: parentIdNum
      };
      
      // Add to parent's subtasks
      parentTask.subtasks.push(newSubtask);
      
      log('info', `Created new subtask ${parentIdNum}.${newSubtaskId}`);
    } else {
      throw new Error('Either existingTaskId or newSubtaskData must be provided');
    }
    
    // Write the updated tasks back to the file
    writeJSON(tasksPath, data);
    
    // Generate task files if requested
    if (generateFiles) {
      log('info', 'Regenerating task files...');
      await generateTaskFiles(tasksPath, path.dirname(tasksPath));
    }
    
    return newSubtask;
  } catch (error) {
    log('error', `Error adding subtask: ${error.message}`);
    throw error;
  }
}

/**
 * Check if a task is dependent on another task (directly or indirectly)
 * Used to prevent circular dependencies
 * @param {Array} allTasks - Array of all tasks
 * @param {Object} task - The task to check
 * @param {number} targetTaskId - The task ID to check dependency against
 * @returns {boolean} Whether the task depends on the target task
 */
function isTaskDependentOn(allTasks, task, targetTaskId) {
  // If the task is a subtask, check if its parent is the target
  if (task.parentTaskId === targetTaskId) {
    return true;
  }
  
  // Check direct dependencies
  if (task.dependencies && task.dependencies.includes(targetTaskId)) {
    return true;
  }
  
  // Check dependencies of dependencies (recursive)
  if (task.dependencies) {
    for (const depId of task.dependencies) {
      const depTask = allTasks.find(t => t.id === depId);
      if (depTask && isTaskDependentOn(allTasks, depTask, targetTaskId)) {
        return true;
      }
    }
  }
  
  // Check subtasks for dependencies
  if (task.subtasks) {
    for (const subtask of task.subtasks) {
      if (isTaskDependentOn(allTasks, subtask, targetTaskId)) {
        return true;
      }
    }
  }
  
  return false;
}

/**
 * Remove a subtask from its parent task
 * @param {string} tasksPath - Path to the tasks.json file
 * @param {string} subtaskId - ID of the subtask to remove in format "parentId.subtaskId"
 * @param {boolean} convertToTask - Whether to convert the subtask to a standalone task
 * @param {boolean} generateFiles - Whether to regenerate task files after removing the subtask
 * @returns {Object|null} The removed subtask if convertToTask is true, otherwise null
 */
async function removeSubtask(tasksPath, subtaskId, convertToTask = false, generateFiles = true) {
  try {
    log('info', `Removing subtask ${subtaskId}...`);
    
    // Read the existing tasks
    const data = readJSON(tasksPath);
    if (!data || !data.tasks) {
      throw new Error(`Invalid or missing tasks file at ${tasksPath}`);
    }
    
    // Parse the subtask ID (format: "parentId.subtaskId")
    if (!subtaskId.includes('.')) {
      throw new Error(`Invalid subtask ID format: ${subtaskId}. Expected format: "parentId.subtaskId"`);
    }
    
    const [parentIdStr, subtaskIdStr] = subtaskId.split('.');
    const parentId = parseInt(parentIdStr, 10);
    const subtaskIdNum = parseInt(subtaskIdStr, 10);
    
    // Find the parent task
    const parentTask = data.tasks.find(t => t.id === parentId);
    if (!parentTask) {
      throw new Error(`Parent task with ID ${parentId} not found`);
    }
    
    // Check if parent has subtasks
    if (!parentTask.subtasks || parentTask.subtasks.length === 0) {
      throw new Error(`Parent task ${parentId} has no subtasks`);
    }
    
    // Find the subtask to remove
    const subtaskIndex = parentTask.subtasks.findIndex(st => st.id === subtaskIdNum);
    if (subtaskIndex === -1) {
      throw new Error(`Subtask ${subtaskId} not found`);
    }
    
    // Get a copy of the subtask before removing it
    const removedSubtask = { ...parentTask.subtasks[subtaskIndex] };
    
    // Remove the subtask from the parent
    parentTask.subtasks.splice(subtaskIndex, 1);
    
    // If parent has no more subtasks, remove the subtasks array
    if (parentTask.subtasks.length === 0) {
      delete parentTask.subtasks;
    }
    
    let convertedTask = null;
    
    // Convert the subtask to a standalone task if requested
    if (convertToTask) {
      log('info', `Converting subtask ${subtaskId} to a standalone task...`);
      
      // Find the highest task ID to determine the next ID
      const highestId = Math.max(...data.tasks.map(t => t.id));
      const newTaskId = highestId + 1;
      
      // Create the new task from the subtask
      convertedTask = {
        id: newTaskId,
        title: removedSubtask.title,
        description: removedSubtask.description || '',
        details: removedSubtask.details || '',
        status: removedSubtask.status || 'pending',
        dependencies: removedSubtask.dependencies || [],
        priority: parentTask.priority || 'medium' // Inherit priority from parent
      };
      
      // Add the parent task as a dependency if not already present
      if (!convertedTask.dependencies.includes(parentId)) {
        convertedTask.dependencies.push(parentId);
      }
      
      // Add the converted task to the tasks array
      data.tasks.push(convertedTask);
      
      log('info', `Created new task ${newTaskId} from subtask ${subtaskId}`);
    } else {
      log('info', `Subtask ${subtaskId} deleted`);
    }
    
    // Write the updated tasks back to the file
    writeJSON(tasksPath, data);
    
    // Generate task files if requested
    if (generateFiles) {
      log('info', 'Regenerating task files...');
      await generateTaskFiles(tasksPath, path.dirname(tasksPath));
    }
    
    return convertedTask;
  } catch (error) {
    log('error', `Error removing subtask: ${error.message}`);
    throw error;
  }
}

// Export task manager functions
export {
  parsePRD,
  updateTasks,
  generateTaskFiles,
  setTaskStatus,
  updateSingleTaskStatus,
  listTasks,
  expandTask,
  expandAllTasks,
  clearSubtasks,
  addTask,
  addSubtask,
  removeSubtask,
  findNextTask,
  analyzeTaskComplexity,
}; 