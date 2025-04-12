/**
 * ai-services.js
 * AI service interactions for the Task Master CLI
 */

// NOTE/TODO: Include the beta header output-128k-2025-02-19 in your API request to increase the maximum output token length to 128k tokens for Claude 3.7 Sonnet.
// ^^^ This note is specific to Anthropic and can be removed or updated for Gemini model capabilities.

import { GoogleGenerativeAI } from "@google/generative-ai";
import OpenAI from 'openai';
import dotenv from 'dotenv';
import { CONFIG, log, sanitizePrompt } from './utils.js';
import { startLoadingIndicator, stopLoadingIndicator } from './ui.js';
import chalk from 'chalk';
import path from 'path';
import fs from 'fs';
import { readJSON, writeJSON } from './utils.js';

// Load environment variables
dotenv.config();

// Configure Google Generative AI client
if (!process.env.GEMINI_API_KEY) {
  // Log error and exit or throw if the key is essential for startup
  log('error', 'GEMINI_API_KEY environment variable is missing. Please set it in your .env file.');
  // Decide on behavior: throw error, exit, or allow proceeding if AI isn't immediately needed.
  // For now, let's log and potentially let other non-AI commands work.
  // Consider throwing an error if AI is critical: throw new Error("GEMINI_API_KEY is not set.");
}
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || ''); // Initialize, even if key is missing, to avoid crashing on import. Check key before use.
let generativeModel; // Declare model variable

// Function to get the configured model, ensuring API key exists
function getGenerativeModel() {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is missing. Cannot initialize AI model.");
  }
  if (!generativeModel) {
    // Use the model name from CONFIG, assuming it will be updated e.g., 'gemini-1.5-flash'
    generativeModel = genAI.getGenerativeModel({ model: CONFIG.model });
  }
  return generativeModel;
}

// Lazy-loaded Perplexity client
let perplexity = null;

/**
 * Get or initialize the Perplexity client
 * @returns {OpenAI} Perplexity client
 */
function getPerplexityClient() {
  if (!perplexity) {
    if (!process.env.PERPLEXITY_API_KEY) {
      throw new Error("PERPLEXITY_API_KEY environment variable is missing. Set it to use research-backed features.");
    }
    perplexity = new OpenAI({
      apiKey: process.env.PERPLEXITY_API_KEY,
      baseURL: 'https://api.perplexity.ai',
    });
  }
  return perplexity;
}

/**
 * Handle API errors with user-friendly messages
 * @param {Error} error - The error from the API call
 * @returns {string} User-friendly error message
 */
function handleApiError(error, serviceName = "AI Service") { // Renamed and made generic
  // --- DEBUGGING: Print the original error object --- 
  console.error(`[DEBUG] Raw error object received in handleApiError for ${serviceName}:`);
  console.error(error);
  // --- END DEBUGGING ---

  // Basic error checking for Gemini (structure might vary)
  // Gemini errors might not have a standard 'type' like Anthropic. Check message content.
  if (error.message?.includes('API key not valid')) {
      return `${serviceName} Error: Invalid API Key. Please check your GEMINI_API_KEY.`;
  }
  if (error.message?.includes('quota')) {
      return `${serviceName} Error: Rate limit or quota exceeded. Please check your usage limits or wait and try again.`;
  }
  if (error.message?.toLowerCase().includes('timeout')) {
    return `The request to ${serviceName} timed out. Please try again.`;
  }
  if (error.message?.toLowerCase().includes('network') || error.message?.toLowerCase().includes('fetch')) {
    return `There was a network error connecting to ${serviceName}. Please check your internet connection and try again.`;
  }
  if (error.status === 400) {
     return `${serviceName} Error: Bad Request (400). The input may be malformed or invalid. Details: ${error.message}`;
  }
   if (error.status === 500) {
     return `${serviceName} Error: Internal Server Error (500). The service may be temporarily unavailable. Details: ${error.message}`;
   }

  // Default error message
  log('error', `Raw error from ${serviceName}:`, error); // Log raw error for debugging
  return `Error communicating with ${serviceName}: ${error.message || 'An unknown error occurred.'}`;
}

/**
 * Call the Generative AI model to generate tasks from a PRD
 * @param {string} prdContent - PRD content
 * @param {string} prdPath - Path to the PRD file
 * @param {number} numTasks - Number of tasks to generate
 * @param {number} retryCount - Retry count
 * @returns {Promise<Object>} AI model's response
 */
async function callGenerativeAI(prdContent, prdPath, numTasks, retryCount = 0) { // Renamed function
  // --- DEBUGGING --- 
  console.log(`[DEBUG] Attempting to call Generative AI.`);
  console.log(`[DEBUG] GEMINI_API_KEY loaded: ${process.env.GEMINI_API_KEY ? 'Exists (partially hidden)' : 'MISSING!'}`); // Check if key exists
  console.log(`[DEBUG] Model from CONFIG: ${CONFIG.model}`);
  // --- END DEBUGGING ---

  if (!process.env.GEMINI_API_KEY) {
     const errorMsg = "GEMINI_API_KEY environment variable is missing. Cannot generate tasks.";
     log('error', errorMsg);
     console.error(chalk.red(errorMsg));
     throw new Error(errorMsg);
  }
  try {
    log('info', 'Calling Generative AI...');
    const model = getGenerativeModel();

    // Combine system prompt and user prompt for Gemini
    // Gemini works well with direct instructions in the user prompt.
    const combinedPrompt = `You are an AI assistant helping to break down a Product Requirements Document (PRD) into a set of sequential development tasks.
Your goal is to create ${numTasks} well-structured, actionable development tasks based on the PRD provided.

Each task should follow this JSON structure:
{
  "id": number,
  "title": string,
  "description": string,
  "status": "pending",
  "dependencies": number[] (IDs of tasks this depends on),
  "priority": "high" | "medium" | "low",
  "details": string (implementation details),
  "testStrategy": string (validation approach)
}

Guidelines:
1. Create exactly ${numTasks} tasks, numbered from 1 to ${numTasks}
2. Each task should be atomic and focused on a single responsibility
3. Order tasks logically - consider dependencies and implementation sequence
4. Early tasks should focus on setup, core functionality first, then advanced features
5. Include clear validation/testing approach for each task
6. Set appropriate dependency IDs (a task can only depend on tasks with lower IDs)
7. Assign priority (high/medium/low) based on criticality and dependency order
8. Include detailed implementation guidance in the "details" field

Expected output format:
{
  "tasks": [
    {
      "id": 1,
      "title": "Setup Project Repository",
      "description": "...",
      "dependencies": [],
      "priority": "high",
      "details": "...",
      "testStrategy": "..."
    },
    {
      "id": 2,
      "title": "...",
      "description": "...",
      "dependencies": [1],
      "priority": "medium",
      "details": "...",
      "testStrategy": "..."
    }
  ],
  "metadata": {
    "projectName": "PRD Implementation",
    "totalTasks": ${numTasks},
    "sourceFile": "${prdPath}",
    "generatedAt": "YYYY-MM-DD"
  }
}

Important: Your response must be valid JSON only, starting with '{' and ending with '}', with no additional explanation, comments, markdown formatting, or \`\`\`json markers.

Here's the Product Requirements Document (PRD) to break down into ${numTasks} tasks:
--- PRD START ---
${prdContent}
--- PRD END ---
`;

    // Use streaming request
    return await handleStreamingRequestGemini(combinedPrompt, numTasks, CONFIG.maxTokens, prdPath, retryCount); // Pass retryCount

  } catch (error) {
    const userMessage = handleApiError(error, "Generative AI Service"); // Use updated error handler
    log('error', userMessage);

    // Simplified retry logic - check specific error types if possible, otherwise retry on common network/timeout issues
    if (retryCount < 2 && (
      error.message?.toLowerCase().includes('timeout') ||
      error.message?.toLowerCase().includes('network') ||
      error.message?.toLowerCase().includes('fetch') ||
      error.status === 500 // Retry on internal server errors
    )) {
      const waitTime = (retryCount + 1) * 5000; // 5s, then 10s
      log('info', `Waiting ${waitTime / 1000} seconds before retry ${retryCount + 1}/2...`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      // Pass the original PRD content, not the combined prompt, to the retry call
      return await callGenerativeAI(prdContent, prdPath, numTasks, retryCount + 1);
    } else {
      console.error(chalk.red(userMessage));
      if (CONFIG.debug) {
        log('debug', 'Full error:', error);
      }
      throw new Error(userMessage);
    }
  }
}

/**
 * Handle streaming request to the Generative AI model
 * @param {string} prompt - The combined prompt for the AI
 * @param {number} numTasks - Expected number of tasks (for processing context)
 * @param {number} maxTokens - Maximum output tokens
 * @param {string} prdPath - Path to the PRD file (for metadata)
 * @param {number} retryCount - Current retry count (for passing to processor)
 * @returns {Promise<Object>} AI model's processed response
 */
async function handleStreamingRequestGemini(prompt, numTasks, maxTokens, prdPath, retryCount) { // Renamed function
  const loadingIndicator = startLoadingIndicator('Generating tasks from PRD...');
  let responseText = '';
  let streamingInterval = null;
  const model = getGenerativeModel(); // Get model instance

  try {
    // Define the request structure explicitly
    const generationConfig = {
      maxOutputTokens: maxTokens, // Use the maxTokens parameter passed to the function
      temperature: CONFIG.temperature,
      // Add other generation config like topP, topK if needed
    };
    const contents = [{ role: "user", parts: [{ text: prompt }] }];
    const request = {
      contents,
      generationConfig,
      // safetySettings: ... // Add safety settings if needed
    };

    // Use streaming for handling large responses with Gemini, passing the full request object
    log('debug', `Sending request to generateContentStream with model ${CONFIG.model}`);
    const result = await model.generateContentStream(request); // Pass the explicit request object

    // Update loading indicator to show streaming progress
    let dotCount = 0;
    const readline = await import('readline');
    streamingInterval = setInterval(() => {
      readline.cursorTo(process.stdout, 0);
      process.stdout.write(`Receiving streaming response from Generative AI${'.'.repeat(dotCount)}`);
      dotCount = (dotCount + 1) % 4;
    }, 500);

    // Process the stream
    for await (const chunk of result.stream) {
      // Check if chunk and text() function exist
      if (chunk && typeof chunk.text === 'function') {
        try {
          const chunkText = chunk.text(); // Get text from chunk
          responseText += chunkText;
        } catch (e) {
           log('warn', 'Could not extract text from stream chunk:', e);
           // Continue processing other chunks
        }
      } else {
         log('warn', 'Received invalid chunk structure:', chunk);
      }
    }

    if (streamingInterval) clearInterval(streamingInterval);
    stopLoadingIndicator(loadingIndicator);

    log('info', "Completed streaming response from Generative AI API!");

    // Pass necessary context to the processing function
    return processApiResponse(responseText, numTasks, retryCount, prdPath); // Pass prdPath for metadata

  } catch (error) {
    if (streamingInterval) clearInterval(streamingInterval);
    stopLoadingIndicator(loadingIndicator);

    const userMessage = handleApiError(error, "Generative AI Streaming");
    log('error', userMessage);
    console.error(chalk.red(userMessage));

    if (CONFIG.debug) {
      log('debug', 'Full error:', error);
    }

    throw new Error(userMessage); // Re-throw to be caught by the caller for retry logic
  }
}

/**
 * Process the AI model's response
 * @param {string} textContent - Text content from the AI
 * @param {number} numTasks - Expected number of tasks
 * @param {number} retryCount - Current retry count
 * @param {string} prdPath - Path to the PRD file (for metadata generation)
 * @returns {Object} Processed response (parsed JSON)
 */
function processApiResponse(textContent, numTasks, retryCount, prdPath) { // Renamed function, added prdPath
  try {
    // Attempt to parse the JSON response
    // Trim whitespace and remove potential markdown code fences
    let cleanText = textContent.trim();
    if (cleanText.startsWith('```json')) {
      cleanText = cleanText.substring(7);
    }
    if (cleanText.endsWith('```')) {
      cleanText = cleanText.substring(0, cleanText.length - 3);
    }
    cleanText = cleanText.trim(); // Trim again after removing fences

    let jsonStart = cleanText.indexOf('{');
    let jsonEnd = cleanText.lastIndexOf('}');

    if (jsonStart === -1 || jsonEnd === -1 || jsonEnd < jsonStart) {
      throw new Error("Could not find valid JSON in Generative AI's response");
    }
    
    let jsonContent = cleanText.substring(jsonStart, jsonEnd + 1);
    let parsedData = JSON.parse(jsonContent);
    
    // Validate the structure of the generated tasks
    if (!parsedData.tasks || !Array.isArray(parsedData.tasks)) {
      throw new Error("Generative AI's response does not contain a valid tasks array");
    }
    
    // Ensure we have the correct number of tasks
    if (parsedData.tasks.length !== numTasks) {
      log('warn', `Expected ${numTasks} tasks, but received ${parsedData.tasks.length}`);
    }
    
    // Add metadata if missing
    if (!parsedData.metadata) {
      parsedData.metadata = {
        projectName: "PRD Implementation",
        totalTasks: parsedData.tasks.length,
        sourceFile: prdPath,
        generatedAt: new Date().toISOString().split('T')[0]
      };
    }
    
    return parsedData;
  } catch (error) {
    log('error', "Error processing Generative AI's response:", error.message);
    
    // Retry logic
    if (retryCount < 2) {
      log('info', `Retrying to parse response (${retryCount + 1}/2)...`);
      
      // Try again with Generative AI for a cleaner response
      if (retryCount === 1) {
        log('info', "Calling Generative AI again for a cleaner response...");
        return callGenerativeAI(prdContent, prdPath, numTasks, retryCount + 1);
      }
      
      return processApiResponse(textContent, numTasks, retryCount + 1, prdPath);
    } else {
      throw error;
    }
  }
}

/**
 * Generate subtasks for a task
 * @param {Object} task - Task to generate subtasks for
 * @param {number} numSubtasks - Number of subtasks to generate
 * @param {number} nextSubtaskId - Next subtask ID
 * @param {string} additionalContext - Additional context
 * @returns {Promise<Array>} Generated subtasks
 */
async function generateSubtasks(task, numSubtasks, nextSubtaskId, additionalContext = '') {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is missing. Cannot generate subtasks.");
  }
  const loadingIndicator = startLoadingIndicator(`Generating ${numSubtasks} subtasks for task ${task.id}...`);
  const model = getGenerativeModel();

  try {
    // Construct the prompt for Gemini
    const subtaskPrompt = `
You are an AI assistant tasked with breaking down a larger development task into smaller, manageable subtasks.
The parent task is:
Title: ${task.title}
Description: ${task.description}
Details: ${task.details || 'None provided'}
Priority: ${task.priority}
Dependencies: ${task.dependencies?.join(', ') || 'None'}

Generate exactly ${numSubtasks} subtasks for this parent task, starting with ID ${nextSubtaskId}.
Each subtask should be a specific, actionable step towards completing the parent task.
Order the subtasks logically. Subtasks can depend on preceding subtasks within this list (use their sequential ID like ${nextSubtaskId}, ${nextSubtaskId + 1}, ...).

${additionalContext ? `Additional context to consider: ${additionalContext}\n` : ''}
Output Format:
Provide the subtasks as a JSON array ONLY. Do not include explanations, markdown formatting, or \`\`\`json markers.
Each subtask object in the array must follow this structure:
{
  "id": number (sequential, starting from ${nextSubtaskId}),
  "title": string (concise subtask title),
  "description": string (brief description of the subtask),
  "status": "pending",
  "dependencies": number[] (IDs of preceding subtasks it depends on, e.g., [${nextSubtaskId}] if it depends on the first),
  "priority": "${task.priority}",
  "details": string (specific implementation steps for this subtask),
  "testStrategy": string (how to verify this subtask is done)
}

Example of expected JSON output format (for 3 subtasks starting at ID ${nextSubtaskId}):
[
  { "id": ${nextSubtaskId}, "title": "Subtask 1 Title", "description": "...", "status": "pending", "dependencies": [], "priority": "${task.priority}", "details": "...", "testStrategy": "..." },
  { "id": ${nextSubtaskId + 1}, "title": "Subtask 2 Title", "description": "...", "status": "pending", "dependencies": [${nextSubtaskId}], "priority": "${task.priority}", "details": "...", "testStrategy": "..." },
  { "id": ${nextSubtaskId + 2}, "title": "Subtask 3 Title", "description": "...", "status": "pending", "dependencies": [${nextSubtaskId + 1}], "priority": "${task.priority}", "details": "...", "testStrategy": "..." }
]

Generate the JSON array now for task ${task.id}.
`;

    log('debug', `Generating subtasks for task ${task.id} with prompt:\n${subtaskPrompt}`);

    // Use generateContent for a single, non-streaming response
    const result = await model.generateContent(subtaskPrompt);
    const response = result.response; // Access the response object directly

    stopLoadingIndicator(loadingIndicator);

    if (!response) {
       throw new Error('No response received from the AI model.');
    }

    const responseText = response.text(); // Get the text content
    log('info', `Received subtask generation response for task ${task.id}.`);
    log('debug', `Raw response text for subtasks: ${responseText}`);

    // Parse the generated subtasks
    const generatedSubtasks = parseSubtasksFromText(responseText, nextSubtaskId, numSubtasks, task.id);
    return generatedSubtasks;

  } catch (error) {
    stopLoadingIndicator(loadingIndicator);
    const userMessage = handleApiError(error, "Generative AI Subtask Generation");
    log('error', `Error generating subtasks for task ${task.id}: ${userMessage}`);
    console.error(chalk.red(`Failed to generate subtasks for task ${task.id}: ${userMessage}`));
    if (CONFIG.debug) {
      log('debug', 'Full error details:', error);
    }
    // Return empty array or re-throw, depending on desired behavior on failure
    return []; // Return empty array to avoid crashing the expand process
    // OR: throw new Error(userMessage);
  }
}

/**
 * Parse subtasks from Claude's response text
 * @param {string} text - Raw text response from the AI
 * @param {number} startId - The expected starting ID for subtasks
 * @param {number} expectedCount - The expected number of subtasks
 * @param {number} parentTaskId - The ID of the parent task (for logging/context)
 * @returns {Array<Object>} Parsed subtasks
 */
function parseSubtasksFromText(text, startId, expectedCount, parentTaskId) {
  try {
    log('debug', `Attempting to parse ${expectedCount} subtasks for parent ${parentTaskId} starting with ID ${startId}. Raw text length: ${text?.length || 0}`); // Added length log
    // Clean the text: remove potential markdown fences and trim whitespace
    let cleanText = text?.trim() || ''; // Handle potential null/undefined text
    if (!cleanText) {
        throw new Error("Received empty text response from AI.");
    }

    if (cleanText.startsWith('```json')) {
      cleanText = cleanText.substring(7).trim();
    } else if (cleanText.startsWith('```')) {
        // Handle cases where only ``` is present without 'json'
        cleanText = cleanText.substring(3).trim();
    }

    if (cleanText.endsWith('```')) {
      cleanText = cleanText.substring(0, cleanText.length - 3).trim();
    }

    // Ensure it starts with '[' and ends with ']' for a JSON array
    if (!cleanText.startsWith('[') || !cleanText.endsWith(']')) {
       log('warn', `Response for parent ${parentTaskId} does not appear to be a valid JSON array. Attempting to find array within text.`);
       const startIndex = cleanText.indexOf('[');
       const endIndex = cleanText.lastIndexOf(']');
       if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
           const extractedArray = cleanText.substring(startIndex, endIndex + 1);
           log('debug', `Extracted potential JSON array: ${extractedArray.substring(0, 100)}...`); // Log start of extracted text
           // Basic validation of extracted content
           try {
               JSON.parse(extractedArray); // Try parsing the extracted part
               cleanText = extractedArray; // Use extracted part if it's likely JSON
           } catch (parseError) {
                log('error', `Extracted text segment for parent ${parentTaskId} is not valid JSON: ${parseError.message}`);
                throw new Error("Response does not contain a recognizable JSON array structure.");
           }
       } else {
           throw new Error("Response does not contain a recognizable JSON array structure.");
       }
    }

    const parsed = JSON.parse(cleanText);

    if (!Array.isArray(parsed)) {
      throw new Error("Parsed response is not an array.");
    }

    // Validate and potentially adjust subtask structure
    const validatedSubtasks = parsed.map((subtask, index) => {
      const expectedId = startId + index;
      if (typeof subtask !== 'object' || subtask === null) {
        log('warn', `Invalid subtask structure at index ${index} for parent ${parentTaskId}. Skipping.`);
        return null; // Skip invalid entries
      }
      
      // Basic validation/defaulting
      return {
        id: typeof subtask.id === 'number' ? subtask.id : expectedId, // Use expected ID if missing/invalid
        title: subtask.title || `Subtask ${expectedId}`,
        description: subtask.description || 'No description provided.',
        status: 'pending',
        dependencies: Array.isArray(subtask.dependencies) ? subtask.dependencies.filter(dep => typeof dep === 'number') : [],
        priority: subtask.priority || 'medium', // Default priority if missing
        details: subtask.details || 'No details provided.',
        testStrategy: subtask.testStrategy || 'Manual verification.'
      };
    }).filter(subtask => subtask !== null); // Remove skipped entries

    if (validatedSubtasks.length !== expectedCount) {
      log('warn', `Expected ${expectedCount} subtasks for parent ${parentTaskId}, but parsed ${validatedSubtasks.length}.`);
      // Decide whether to proceed with parsed tasks or throw error
    }
     if (validatedSubtasks.length === 0 && expectedCount > 0) {
         throw new Error("Failed to parse any valid subtasks from the response.");
     }

    log('info', `Successfully parsed ${validatedSubtasks.length} subtasks for parent task ${parentTaskId}.`);
    return validatedSubtasks;

  } catch (error) {
    log('error', `Failed to parse subtasks JSON for parent ${parentTaskId}: ${error.message}`);
    log('debug', `Raw text causing parsing error for parent ${parentTaskId}: ${text}`); // Log raw text on error
    // Return empty array or re-throw, depending on desired error handling
    // Returning empty allows caller to handle potentially partial success or failure gracefully
    return [];
  }
}

/**
 * Generate a prompt for complexity analysis
 * @param {Object} tasksData - Tasks data object containing tasks array
 * @returns {string} Generated prompt
 */
function generateComplexityAnalysisPrompt(tasksData) {
  // Ensure CONFIG.defaultSubtasks is treated as a number
  const defaultSubtasks = Number(CONFIG.defaultSubtasks) || 3;
  const minSubtasks = Math.max(2, defaultSubtasks - 1); // Ensure min is at least 2
  const maxSubtasks = Math.min(10, defaultSubtasks + 2); // Ensure max is reasonable

  return `Analyze the complexity of the following software development tasks and provide recommendations for subtask breakdown.

${tasksData.tasks.map(task => `
--- Task Start ---
Task ID: ${task.id}
Title: ${task.title}
Description: ${sanitizePrompt(task.description)}
Details: ${sanitizePrompt(task.details)}
Dependencies: ${JSON.stringify(task.dependencies || [])}
Priority: ${task.priority || 'medium'}
--- Task End ---
`).join('\n')}

Analyze each task based on its description, details, and dependencies. Return a JSON array ONLY, containing one object per task analyzed.
Each object in the array must follow this structure EXACTLY:
[
  {
    "taskId": number (Must match the original Task ID),
    "taskTitle": string (Must match the original Task Title),
    "complexityScore": number (Estimate complexity on a scale of 1-10, where 1 is trivial and 10 is very complex),
    "recommendedSubtasks": number (Suggest a number of subtasks between ${minSubtasks} and ${maxSubtasks} needed to break this down effectively. Base this on complexity.),
    "expansionPrompt": string (Create a concise, specific prompt suffix to guide an AI in generating useful subtasks for THIS task. Focus on the core challenge or goal. Example: "focusing on database schema design and migration scripts."),
    "reasoning": string (Briefly justify the complexity score and subtask recommendation. Mention key factors like ambiguity, dependencies, scope, etc.)
  }
  // ... include one object for EACH task provided above ...
]

IMPORTANT:
- Respond with ONLY the valid JSON array. Do not include any explanatory text before or after the array.
- Ensure every task provided in the input is included in the output array with the correct "taskId".
- The "recommendedSubtasks" number must be within the range [${minSubtasks}, ${maxSubtasks}].
- The "expansionPrompt" should be tailored to the specific task.
`;
}

/**
 * Analyzes task complexity using an AI model and generates expansion recommendations.
 * @param {object} options - Options for the analysis.
 * @param {string} [options.file='tasks/tasks.json'] - Path to the tasks file.
 * @param {string} [options.output='scripts/task-complexity-report.json'] - Path to save the report.
 * @param {string} [options.model] - Override the default AI model.
 * @param {number} [options.threshold=5] - Minimum complexity score for expansion recommendation.
 * @param {boolean} [options.research=false] - Use Perplexity AI for research-backed analysis.
 * @returns {Promise<object>} - The generated complexity report object.
 */
async function analyzeTaskComplexity(options = {}) {
  const tasksPath = options.file || 'tasks/tasks.json';
  const outputPath = options.output || 'scripts/task-complexity-report.json';
  const modelOverride = options.model;
  const thresholdScore = parseFloat(options.threshold || '5');
  const useResearch = options.research || false;

  log('info', `Analyzing task complexity from: ${tasksPath}`);
  log('info', `Output report will be saved to: ${outputPath}`);
  if (useResearch) {
    log('info', 'Using Perplexity AI for research-backed complexity analysis');
  } else {
    log('info', `Using Generative AI model: ${modelOverride || CONFIG.model}`);
  }

  let loadingIndicator = null;
  let streamingInterval = null; // Needed for Gemini streaming indication (though not using streaming here ideally)

  try {
    // Read tasks.json
    log('info', `Reading tasks from ${tasksPath}...`);
    const tasksData = readJSON(tasksPath);
    if (!tasksData || !tasksData.tasks || !Array.isArray(tasksData.tasks) || tasksData.tasks.length === 0) {
      throw new Error(`No tasks found or tasks array is invalid in ${tasksPath}`);
    }
    log('info', `Found ${tasksData.tasks.length} tasks to analyze.`);

    // Prepare the prompt for the LLM
    const prompt = generateComplexityAnalysisPrompt(tasksData);
    log('debug', 'Generated complexity analysis prompt.');

    // Start loading indicator
    loadingIndicator = startLoadingIndicator('Calling AI to analyze task complexity...');

    let fullResponse = '';

    if (useResearch) {
      // --- Use Perplexity --- 
      try {
        log('info', 'Calling Perplexity AI for complexity analysis...');
        const perplexityClient = getPerplexityClient();
        const researchPrompt = `You are a technical analysis AI. Analyze the provided tasks based on the instructions. CRITICAL: Respond ONLY with the valid JSON array, no explanations or markdown. ${prompt}`;
        const result = await perplexityClient.chat.completions.create({
          model: process.env.PERPLEXITY_MODEL || 'llama-3-sonar-large-32k-online',
          messages: [
            { role: "system", content: "You are a technical analysis AI that only responds with clean, valid JSON." },
            { role: "user", content: researchPrompt }
          ],
          temperature: 0.1, // Low temperature for factual analysis
          max_tokens: CONFIG.maxTokens, // Use configured max tokens
        });
        fullResponse = result?.choices?.[0]?.message?.content || '';
        log('info', 'Received response from Perplexity AI.');
      } catch (perplexityError) {
        stopLoadingIndicator(loadingIndicator);
        log('error', `Perplexity AI call failed: ${perplexityError.message}`);
        // Decide if fallback to Gemini is desired or just throw error
        throw new Error(`Perplexity AI analysis failed: ${handleApiError(perplexityError, 'Perplexity')}`);
      }

    } else {
      // --- Use Gemini --- 
      try {
        log('info', `Calling Generative AI model (${modelOverride || CONFIG.model}) for complexity analysis...`);
        const model = genAI.getGenerativeModel({ model: modelOverride || CONFIG.model }); // Use override if provided
        // GenerateContent is better for single JSON responses than streaming
        const result = await model.generateContent(prompt);
         if (!result || !result.response || typeof result.response.text !== 'function') {
            throw new Error('Invalid response structure received from Gemini API.');
        }
        fullResponse = result.response.text();
        log('info', 'Received response from Generative AI.');
      } catch (geminiError) {
        stopLoadingIndicator(loadingIndicator);
        log('error', `Generative AI call failed: ${geminiError.message}`);
        throw new Error(`Generative AI analysis failed: ${handleApiError(geminiError, 'Generative AI')}`);
      }
    }

    stopLoadingIndicator(loadingIndicator);
    loadingIndicator = null; // Reset indicator

    log('debug', `Raw AI response length: ${fullResponse.length}`);
    log('debug', `Raw AI response (first 200): ${fullResponse.substring(0, 200)}`);

    // Parse the JSON response
    log('info', `Parsing complexity analysis JSON...`);
    let complexityAnalysis;
    try {
      // Clean up the response to ensure it's valid JSON
      let cleanedResponse = fullResponse.trim();
      const jsonMatch = cleanedResponse.match(/```(?:json)?\s*(\[\s\S]*?\])\s*```/s) || cleanedResponse.match(/(\[\s\S]*?\])/s);
      
      if (!jsonMatch || !jsonMatch[1]) {
           log('warn', 'Could not find a clear JSON array in the response. Attempting direct parse.');
           // Attempt direct parse if no clear array found
           if (cleanedResponse.startsWith('[') && cleanedResponse.endsWith(']')) {
                // Looks like an array, try parsing directly
           } else if (cleanedResponse.startsWith('{')) {
                // Maybe it's an object containing the array?
                const potentialObj = JSON.parse(cleanedResponse); 
                if (Array.isArray(potentialObj.analysis)) cleanedResponse = JSON.stringify(potentialObj.analysis);
                else if (Array.isArray(potentialObj.tasks)) cleanedResponse = JSON.stringify(potentialObj.tasks);
                else throw new Error('Response is an object but does not contain a recognizable task analysis array.');
           } else {
                throw new Error("AI response does not appear to be a JSON array or object containing one.");
           }
      } else {
           cleanedResponse = jsonMatch[1];
           log('debug', 'Extracted JSON array from response.');
      }

      complexityAnalysis = JSON.parse(cleanedResponse);

      if (!Array.isArray(complexityAnalysis)) {
        throw new Error("Parsed response is not a valid JSON array.");
      }

      log('info', `Successfully parsed ${complexityAnalysis.length} analysis entries.`);

      // Optional: Validate structure of each analysis object
      complexityAnalysis = complexityAnalysis.filter(entry => entry && typeof entry.taskId === 'number'); 
      log('info', `Filtered down to ${complexityAnalysis.length} valid entries after basic validation.`);

    } catch (parseError) {
      log('error', `Failed to parse AI response as JSON: ${parseError.message}`);
      log('debug', `Raw response causing parse error: ${fullResponse}`);
      throw new Error(`Failed to parse AI analysis response: ${parseError.message}`);
    }

    // Create the final report object
    const report = {
      meta: {
        generatedAt: new Date().toISOString(),
        tasksAnalyzed: tasksData.tasks.length, // Total tasks from input
        analysisEntries: complexityAnalysis.length, // Tasks successfully analyzed
        thresholdScore: thresholdScore,
        projectName: tasksData.meta?.projectName || CONFIG.projectName,
        tasksFile: tasksPath,
        reportFile: outputPath,
        modelUsed: useResearch ? `Perplexity (${process.env.PERPLEXITY_MODEL || 'default'})` : (modelOverride || CONFIG.model),
        usedResearch: useResearch
      },
      complexityAnalysis: complexityAnalysis
    };

    // Write the report to file
    log('info', `Writing complexity report to ${outputPath}...`);
    // Ensure directory exists before writing
    const outputDir = path.dirname(outputPath);
     if (!fs.existsSync(outputDir)){
         fs.mkdirSync(outputDir, { recursive: true });
         log('debug', `Created directory: ${outputDir}`);
     }
    writeJSON(outputPath, report);

    log('success', `Task complexity analysis complete. Report written to ${outputPath}`);

    // Display summary (optional)
    const highComplexity = complexityAnalysis.filter(t => t.complexityScore >= 8).length;
    const mediumComplexity = complexityAnalysis.filter(t => t.complexityScore >= 5 && t.complexityScore < 8).length;
    const lowComplexity = complexityAnalysis.filter(t => t.complexityScore < 5).length;
    console.log(chalk.cyan('\nComplexity Analysis Summary:'));
    console.log(`  High complexity (>=8): ${highComplexity}`);
    console.log(`  Medium complexity (5-7): ${mediumComplexity}`);
    console.log(`  Low complexity (<5): ${lowComplexity}`);
    console.log(`  Total tasks analyzed: ${complexityAnalysis.length}/${tasksData.tasks.length}`);

    return report; // Return the generated report object

  } catch (error) {
    // Ensure loading indicator is stopped in case of error
    if (loadingIndicator) stopLoadingIndicator(loadingIndicator);
    log('error', `Error during complexity analysis: ${error.message}`);
    console.error(chalk.red(`Error analyzing task complexity: ${error.message}`));
    if (CONFIG.debug && error.stack) {
      log('debug', error.stack);
    }
    // Rethrow or exit depending on desired behavior
    throw error; // Propagate error up
  }
}

/**
 * Generate subtasks with research from Perplexity
 * @param {Object} task - Task to generate subtasks for
 * @param {number} numSubtasks - Number of subtasks to generate
 * @param {number} nextSubtaskId - Next subtask ID
 * @param {string} additionalContext - Additional context
 * @returns {Promise<Array>} Generated subtasks
 */
async function generateSubtasksWithPerplexity(task, numSubtasks = 3, nextSubtaskId = 1, additionalContext = '') {
  let researchLoadingIndicator = null;
  let subtaskLoadingIndicator = null; // Keep track of the generateSubtasks indicator if needed

  try {
    // First, perform research to get context
    log('info', `Researching context for task ${task.id}: ${task.title}`);
    const perplexityClient = getPerplexityClient(); // Use the getter function

    const PERPLEXITY_MODEL = process.env.PERPLEXITY_MODEL || 'llama-3-sonar-large-32k-online'; // Updated default model
    researchLoadingIndicator = startLoadingIndicator('Researching best practices with Perplexity AI...');

    // Formulate research query based on task
    const researchQuery = `For the software development task titled "${task.title}" (Description: "${task.description}"), provide current best practices, relevant libraries or frameworks, potential design patterns, and key implementation approaches. Focus on practical advice and technical considerations. If applicable, include concise code examples or snippets illustrating the concepts.`;

    // Query Perplexity for research
    const researchResponse = await perplexityClient.chat.completions.create({
      model: PERPLEXITY_MODEL,
      messages: [{
        role: 'system',
        content: 'You are a helpful AI assistant providing technical research for software development tasks.'
      }, {
        role: 'user',
        content: researchQuery
      }],
      max_tokens: 1024, // Limit response size
      temperature: 0.2 // Lower temperature for more factual responses
    });

    stopLoadingIndicator(researchLoadingIndicator);
    researchLoadingIndicator = null; // Reset indicator

    const researchResult = researchResponse?.choices?.[0]?.message?.content || "No research result obtained.";
    log('info', 'Perplexity research completed.');
    log('debug', `Perplexity research result for task ${task.id}: ${researchResult}`);

    // Use the research result as additional context for Gemini to generate subtasks
    const combinedContext = `
--- Relevant Research Findings ---
${researchResult}
--- End Research Findings ---

${additionalContext ? `--- Additional User Context ---\n${additionalContext}\n--- End User Context ---` : ''}
`;

    // Now generate subtasks using the main generateSubtasks function, passing the combined context
    log('info', `Generating research-backed subtasks for task ${task.id} using Generative AI...`);
    // The generateSubtasks function already includes its own loading indicator.
    return await generateSubtasks(task, numSubtasks, nextSubtaskId, combinedContext);

  } catch (error) {
    // Ensure indicators are stopped if they were started
    if (researchLoadingIndicator) stopLoadingIndicator(researchLoadingIndicator);
    // generateSubtasks handles its own indicator stopping

    const userMessage = handleApiError(error, "Perplexity Research / Subtask Generation");
    log('error', `Error during research-backed subtask generation for task ${task.id}: ${userMessage}`);
    console.error(chalk.red(`Failed research-backed subtask generation for task ${task.id}: ${userMessage}`));

    if (CONFIG.debug) {
      log('debug', 'Full error details during research-backed generation:', error);
    }
    // Return empty array to allow the process to potentially continue
    return [];
  }
} // End of generateSubtasksWithPerplexity function

/**
 * Calls the Generative AI model to update a list of tasks based on a prompt.
 * @param {Array<Object>} tasksToUpdate - The array of task objects to be updated.
 * @param {string} updatePrompt - The prompt describing the required changes.
 * @param {string} systemContext - General system context/instructions for the AI.
 * @returns {Promise<Array<Object>>} - A promise that resolves to the array of updated task objects.
 */
async function callGenerativeAIForUpdate(tasksToUpdate, updatePrompt, systemContext) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is missing. Cannot update tasks.");
  }
  if (!tasksToUpdate || tasksToUpdate.length === 0) {
    log('warn', 'No tasks provided to update.');
    return []; // Return empty if no tasks need updating
  }

  const loadingIndicator = startLoadingIndicator('Calling Generative AI to update tasks...');
  let model; // Declare model variable outside try block
  try {
      model = getGenerativeModel(); // Initialize model inside try block in case getGenerativeModel throws
  } catch (modelError) {
      stopLoadingIndicator(loadingIndicator);
      log('error', `Failed to get generative model: ${modelError.message}`);
      throw new Error(`Failed to initialize AI model for update: ${modelError.message}`);
  }

  try {
    const taskDataString = JSON.stringify(tasksToUpdate, null, 2);

    // Construct the prompt for Gemini, specifically for updating tasks
    const combinedPrompt = `${systemContext}\n\nHere are the tasks that need updating based on new context:\n\`\`\`json\n${taskDataString}\n\`\`\`\n\nThe new context or required change is:\n"${updatePrompt}"\n\nPlease review each task provided above. Update their fields (title, description, details, testStrategy, etc.) to accurately reflect the new context. Maintain the original task IDs, statuses, dependencies, and priorities unless the context explicitly requires changing them.\n\nIMPORTANT: Respond ONLY with the complete, updated list of tasks as a valid JSON array. The array should contain all the tasks provided, in the same order, but with the necessary modifications applied. Do not include any explanatory text, markdown formatting, or code block markers before or after the JSON array.\n`;

    log('debug', `Sending update request to Generative AI model: ${CONFIG.model}`);

    // Use generateContent for a single JSON response
    const result = await model.generateContent(combinedPrompt);

    if (!result || !result.response || typeof result.response.text !== 'function') {
      stopLoadingIndicator(loadingIndicator); // Stop indicator before throwing
      throw new Error('Invalid response structure received from Gemini API during task update.');
    }
    const responseText = result.response.text();
    stopLoadingIndicator(loadingIndicator);
    log('info', 'Received updated tasks response from Generative AI.');
    log('debug', `Raw update response length: ${responseText.length}`);

    // Parse the JSON response - reuse or adapt the parsing logic from complexity analysis
    log('info', 'Parsing updated tasks JSON...');
    let updatedTasks;
    try {
      let cleanedResponse = responseText.trim();
      let jsonString = null;

      // --- Start: Enhanced JSON Extraction Logic ---
      // Strategy 1: Find the first '[' and last ']'
      const firstBracket = cleanedResponse.indexOf('[');
      const lastBracket = cleanedResponse.lastIndexOf(']');

      if (firstBracket !== -1 && lastBracket !== -1 && lastBracket > firstBracket) {
        jsonString = cleanedResponse.substring(firstBracket, lastBracket + 1);
        log('debug', 'Attempting to parse JSON found between first [ and last ].');
        try {
          updatedTasks = JSON.parse(jsonString);
          if (!Array.isArray(updatedTasks)) {
            log('warn', 'Parsed content between brackets is not an array. Resetting.');
            updatedTasks = null; // Reset if not an array
            jsonString = null;   // Reset jsonString to allow fallback
          }
        } catch (e) {
          log('warn', 'Failed to parse content between first [ and last ]. Trying fallback.');
          updatedTasks = null; // Reset on parse error
          jsonString = null;   // Reset jsonString to allow fallback
        }
      }
      // --- End: Enhanced JSON Extraction Logic ---

      // Strategy 2: Fallback to regex for code blocks if Strategy 1 failed or result wasn't an array
      if (!updatedTasks) { 
        log('debug', 'Fallback: Searching for JSON array within markdown code blocks.');
        const jsonMatch = cleanedResponse.match(/```(?:json)?\s*(\[\s\S]*?\])\s*```/s) || cleanedResponse.match(/(\[\s\S]*?\])/s);

        if (jsonMatch && jsonMatch[1]) {
          jsonString = jsonMatch[1];
          log('debug', 'Extracted JSON array using regex.');
          try {
             updatedTasks = JSON.parse(jsonString);
             if (!Array.isArray(updatedTasks)) {
                log('error', 'Content extracted via regex is not a valid JSON array.');
                throw new Error("Parsed update response (from regex) is not a valid JSON array.");
             }
          } catch (parseError) {
             log('error', `Failed to parse JSON extracted via regex: ${parseError.message}`);
             log('debug', `Content attempted for regex parse: ${jsonString}`);
             throw new Error(`Failed to parse AI task update response (from regex): ${parseError.message}`);
          }
        } else {
          // If no JSON array found by either method
          log('error', 'Could not find a valid JSON array in the AI response using any method.');
          log('debug', `Raw response: ${responseText}`);
          throw new Error("AI response for update does not appear to contain a JSON array.");
        }
      }

      // Basic validation: Check if the number of tasks returned matches the number sent
      if (updatedTasks.length !== tasksToUpdate.length) {
          log('warn', `Number of tasks returned (${updatedTasks.length}) does not match number sent (${tasksToUpdate.length}).`);
          // Decide how to handle this: throw error, try to match by ID, or proceed with caution.
          // For now, we proceed but log a warning.
      }

      // Further validation could involve checking if all original IDs are present.

      log('info', `Successfully parsed ${updatedTasks.length} updated task entries.`);
      return updatedTasks; // Return the array of updated tasks

    } catch (parseError) {
      log('error', `Failed to parse AI update response as JSON: ${parseError.message}`);
      log('debug', `Raw response causing update parse error: ${responseText}`);
      throw new Error(`Failed to parse AI task update response: ${parseError.message}`);
    }

  } catch (error) {
    // Ensure indicator is stopped if it wasn't already
    // Check if loadingIndicator is truthy before stopping
    if (loadingIndicator && typeof stopLoadingIndicator === 'function') {
         try {
            stopLoadingIndicator(loadingIndicator);
         } catch (indicatorError) {
            log('warn', 'Failed to stop loading indicator during error handling.');
         } 
    }
    const userMessage = handleApiError(error, "Generative AI Task Update");
    log('error', `Error updating tasks via AI: ${userMessage}`);
    console.error(chalk.red(`Failed to update tasks using AI: ${userMessage}`));
    if (CONFIG.debug && error.stack) {
      log('debug', error.stack);
    }
    // Rethrow a potentially more informative error if it came from parsing
    if (error.message.startsWith("Failed to parse")) {
        throw error;
    }
    // Otherwise, throw the handled API error
    throw new Error(`AI Task Update failed: ${userMessage}`);
  }
} // End of callGenerativeAIForUpdate function

// Export AI service functions
export {
  callGenerativeAI,
  generateSubtasks,
  generateSubtasksWithPerplexity,
  analyzeTaskComplexity,
  callGenerativeAIForUpdate,
  getPerplexityClient,
  handleApiError,
  handleStreamingRequestGemini,
  processApiResponse,
  parseSubtasksFromText,
  generateComplexityAnalysisPrompt
}; 