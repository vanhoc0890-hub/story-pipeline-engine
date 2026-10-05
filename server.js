require('dotenv').config();

const express = require('express');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');
const { jsonrepair } = require('jsonrepair');

const app = express();

const PORT =
  process.env.PORT ||
  8080;

const MODEL_NAME =
  process.env.GEMINI_MODEL ||
  'gemini-3.5-flash-lite';

const API_KEY =
  process.env.GEMINI_API_KEY;

app.use(cors());

app.use(
  express.json({
    limit: '20mb'
  })
);

let ai = null;

if (API_KEY) {
  ai = new GoogleGenAI({
    apiKey: API_KEY
  });
}

/* =====================================================
   HELPERS
===================================================== */

function cleanText(value) {
  return typeof value === 'string'
    ? value.trim()
    : '';
}

function normalizePreviousOutputs(
  previousOutputs
) {
  if (
    !Array.isArray(
      previousOutputs
    )
  ) {
    return [];
  }

  return previousOutputs
    .map(
      (
        item,
        index
      ) => {
        if (
          typeof item === 'string'
        ) {
          return {
            stepNumber:
              index + 1,
            name:
              `Step ${index + 1}`,
            output:
              item
          };
        }

        if (
          item &&
          typeof item === 'object'
        ) {
          return {
            stepNumber:
              Number(
                item.stepNumber
              ) ||
              index + 1,

            name:
              cleanText(
                item.name
              ) ||
              `Step ${index + 1}`,

            output:
              typeof item.output ===
              'string'
                ? item.output
                : JSON.stringify(
                    item.output ??
                      '',
                    null,
                    2
                  )
          };
        }

        return null;
      }
    )
    .filter(Boolean);
}

function buildPreviousContext(
  previousOutputs
) {
  const normalized =
    normalizePreviousOutputs(
      previousOutputs
    );

  if (!normalized.length) {
    return 'No previous workflow outputs.';
  }

  return normalized
    .map(
      item => {
        return [
          `STEP ${item.stepNumber}: ${item.name}`,
          item.output
        ].join('\n');
      }
    )
    .join(
      '\n\n========================================\n\n'
    );
}

function buildStepPrompt({
  script,
  stepName,
  instruction,
  previousOutputs,
  outputType
}) {
  const previousContext =
    buildPreviousContext(
      previousOutputs
    );

  return `
You are executing one step inside a sequential AI workflow.

IMPORTANT WORKFLOW RULES:

- Follow the CURRENT STEP INSTRUCTION precisely.
- Use the ORIGINAL SCRIPT as the primary source of truth.
- Use PREVIOUS STEP OUTPUTS as accumulated workflow context.
- Do not ignore earlier results when they are relevant.
- Do not invent contradictions to explicit script facts.
- Do not perform later workflow steps unless the current instruction explicitly requests them.
- Return only the result for the CURRENT STEP.
- Keep the output useful for the next workflow step.

==================================================
CURRENT STEP
==================================================

STEP NAME:
${stepName}

INSTRUCTION:
${instruction}

==================================================
REQUESTED OUTPUT TYPE
==================================================

${outputType}

${
  outputType === 'JSON'
    ? `
Return valid JSON only.

Do not use markdown code fences.
Do not include commentary before or after the JSON.
`
    : `
Return plain text only unless the instruction explicitly asks for another presentation format.
`
}

==================================================
PREVIOUS STEP OUTPUTS
==================================================

${previousContext}

==================================================
ORIGINAL SCRIPT
==================================================

${script}
`.trim();
}

function parseJsonOutput(
  rawText
) {
  const text =
    cleanText(rawText);

  if (!text) {
    throw new Error(
      'EMPTY_MODEL_OUTPUT'
    );
  }

  let cleaned =
    text
      .replace(
        /^```json\s*/i,
        ''
      )
      .replace(
        /^```\s*/i,
        ''
      )
      .replace(
        /\s*```$/i,
        ''
      )
      .trim();

  try {
    return JSON.parse(
      cleaned
    );
  } catch {
    try {
      cleaned =
        jsonrepair(
          cleaned
        );

      return JSON.parse(
        cleaned
      );
    } catch {
      throw new Error(
        'JSON_PARSE_FAILED'
      );
    }
  }
}

async function runGeminiStep(
  prompt
) {
  if (!ai) {
    throw new Error(
      'GEMINI_API_KEY_NOT_CONFIGURED'
    );
  }

  const response =
    await ai.models.generateContent({
      model:
        MODEL_NAME,

      contents: [
        {
          role: 'user',
          parts: [
            {
              text: prompt
            }
          ]
        }
      ],

      config: {
        temperature: 0.4
      }
    });

  const text =
    cleanText(
      response?.text
    );

  if (!text) {
    throw new Error(
      'EMPTY_MODEL_OUTPUT'
    );
  }

  return text;
}

/* =====================================================
   HEALTH
===================================================== */

app.get(
  '/health',
  (
    req,
    res
  ) => {
    res.json({
      ok: true,
      service:
        'story-pipeline-engine',
      mode:
        'SEQUENTIAL_DYNAMIC_WORKFLOW_V1',
      model:
        MODEL_NAME,
      geminiConfigured:
        Boolean(API_KEY)
    });
  }
);

/* =====================================================
   RUN ONE WORKFLOW STEP
===================================================== */

app.post(
  '/run-step',
  async (
    req,
    res
  ) => {
    const {
      script,
      stepName,
      instruction,
      previousOutputs,
      outputType = 'TEXT'
    } =
      req.body || {};

    const normalizedScript =
      cleanText(script);

    const normalizedStepName =
      cleanText(
        stepName
      ) ||
      'Untitled Step';

    const normalizedInstruction =
      cleanText(
        instruction
      );

    const normalizedOutputType =
      outputType === 'JSON'
        ? 'JSON'
        : 'TEXT';

    if (!normalizedScript) {
      return res
        .status(400)
        .json({
          ok: false,
          error:
            'SCRIPT_REQUIRED'
        });
    }

    if (
      !normalizedInstruction
    ) {
      return res
        .status(400)
        .json({
          ok: false,
          error:
            'STEP_INSTRUCTION_REQUIRED'
        });
    }

    try {
      const prompt =
        buildStepPrompt({
          script:
            normalizedScript,
          stepName:
            normalizedStepName,
          instruction:
            normalizedInstruction,
          previousOutputs,
          outputType:
            normalizedOutputType
        });

      const rawOutput =
        await runGeminiStep(
          prompt
        );

      let output =
        rawOutput;

      if (
        normalizedOutputType ===
        'JSON'
      ) {
        output =
          parseJsonOutput(
            rawOutput
          );
      }

      return res.json({
        ok: true,

        stepName:
          normalizedStepName,

        outputType:
          normalizedOutputType,

        output
      });
    } catch (error) {
      console.error(
        '[RUN_STEP_FAILED]',
        error
      );

      const message =
        error?.message ||
        'UNKNOWN_ERROR';

      if (
        message.includes(
          '429'
        )
      ) {
        return res
          .status(429)
          .json({
            ok: false,
            error:
              'USAGE_LIMIT_REACHED',
            detail:
              message
          });
      }

      if (
        message ===
        'GEMINI_API_KEY_NOT_CONFIGURED'
      ) {
        return res
          .status(500)
          .json({
            ok: false,
            error:
              'GEMINI_API_KEY_NOT_CONFIGURED'
          });
      }

      if (
        message ===
        'JSON_PARSE_FAILED'
      ) {
        return res
          .status(500)
          .json({
            ok: false,
            error:
              'JSON_PARSE_FAILED'
          });
      }

      return res
        .status(500)
        .json({
          ok: false,
          error:
            'RUN_STEP_FAILED',
          detail:
            message
        });
    }
  }
);

/* =====================================================
   START SERVER
===================================================== */

app.listen(
  PORT,
  () => {
    console.log(
      `Story Pipeline Engine listening on port ${PORT}`
    );
  }
);
