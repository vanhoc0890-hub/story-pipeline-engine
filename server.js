require('dotenv').config();

const express = require('express');
const cors = require('cors');

const { GoogleGenAI } = require('@google/genai');
const { jsonrepair } = require('jsonrepair');

const app = express();

const PORT =
  process.env.PORT || 8080;

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
  ai =
    new GoogleGenAI({
      apiKey: API_KEY
    });
}

/* =====================================================
   BASIC HELPERS
===================================================== */

function cleanText(value) {
  if (
    value === null ||
    value === undefined
  ) {
    return '';
  }

  return String(value).trim();
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

  return previousOutputs.map(
    (item, index) => ({
      stepNumber:
        Number(
          item?.stepNumber
        ) ||
        index + 1,

      name:
        cleanText(
          item?.name
        ) ||
        `Step ${index + 1}`,

      outputType:
        cleanText(
          item?.outputType
        ).toUpperCase(),

      outputDestination:
        cleanText(
          item?.outputDestination
        ).toUpperCase(),

      output:
        item?.output
    })
  );
}

function formatOutputForPrompt(
  output
) {
  if (
    output === null ||
    output === undefined
  ) {
    return '';
  }

  if (
    typeof output ===
    'string'
  ) {
    return output;
  }

  try {
    return JSON.stringify(
      output,
      null,
      2
    );
  } catch {
    return String(
      output
    );
  }
}

function buildPreviousContext(
  previousOutputs
) {
  if (
    !previousOutputs.length
  ) {
    return 'No previous workflow outputs.';
  }

  return previousOutputs
    .map(
      item => [
        `STEP ${item.stepNumber}`,
        `NAME: ${item.name}`,
        `DESTINATION: ${item.outputDestination || 'GENERAL'}`,
        '',
        formatOutputForPrompt(
          item.output
        )
      ].join('\n')
    )
    .join(
      '\n\n========================================\n\n'
    );
}

/* =====================================================
   GEMINI
===================================================== */

async function runGeminiStep(
  prompt,
  temperature = 0.3
) {
  if (!ai) {
    const error =
      new Error(
        'GEMINI_API_KEY_NOT_CONFIGURED'
      );

    error.code =
      'GEMINI_API_KEY_NOT_CONFIGURED';

    throw error;
  }

  const response =
    await ai.models.generateContent({
      model:
        MODEL_NAME,

      contents: [
        {
          role:
            'user',

          parts: [
            {
              text:
                prompt
            }
          ]
        }
      ],

      config: {
        temperature
      }
    });

  const text =
    cleanText(
      response?.text
    );

  if (!text) {
    throw new Error(
      'EMPTY_GEMINI_RESPONSE'
    );
  }

  return text;
}

/* =====================================================
   JSON PARSER
===================================================== */

function stripCodeFence(
  text
) {
  let cleaned =
    cleanText(text);

  cleaned =
    cleaned.replace(
      /^```(?:json)?\s*/i,
      ''
    );

  cleaned =
    cleaned.replace(
      /\s*```$/i,
      ''
    );

  return cleaned.trim();
}

function parseJsonOutput(
  text
) {
  const cleaned =
    stripCodeFence(
      text
    );

  try {
    return JSON.parse(
      cleaned
    );
  } catch {
    // Continue to repair.
  }

  try {
    const repaired =
      jsonrepair(
        cleaned
      );

    return JSON.parse(
      repaired
    );
  } catch (
    error
  ) {
    const parseError =
      new Error(
        'JSON_PARSE_FAILED'
      );

    parseError.detail =
      error?.message ||
      'Unable to parse JSON';

    throw parseError;
  }
}

/* =====================================================
   STANDARD WORKFLOW PROMPT
===================================================== */

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

  const outputRule =
    outputType ===
    'JSON'
      ? `
OUTPUT FORMAT:
Return valid JSON only.
No markdown code fences.
No commentary before or after the JSON.
`
      : `
OUTPUT FORMAT:
Return only the result of the current step.
Use plain text unless the current instruction explicitly requires another text structure.
`;

  return `
You are executing ONE STEP inside a sequential AI story-production workflow.

GENERAL RULES:
- Follow the CURRENT STEP INSTRUCTION precisely.
- The FULL CURRENT SCRIPT is the primary source of truth.
- Previous completed workflow outputs are accumulated context.
- Do not ignore relevant information established by earlier completed steps.
- Do not contradict the current script.
- Do not perform later workflow steps unless explicitly requested.
- Return only the result for the current step.
- Make the result useful for the next workflow step.

CURRENT STEP:
${stepName}

CURRENT STEP INSTRUCTION:
${instruction}

${outputRule}

PREVIOUS COMPLETED WORKFLOW OUTPUTS:
${previousContext}

==================================================
FULL CURRENT SCRIPT
==================================================

${script}
`;
}

/* =====================================================
   CHARACTER LOCK PARSER + NORMALIZER
===================================================== */

function slugify(
  value
) {
  return cleanText(
    value
  )
    .normalize('NFD')
    .replace(
      /[\u0300-\u036f]/g,
      ''
    )
    .toLowerCase()
    .replace(
      /[^a-z0-9]+/g,
      '_'
    )
    .replace(
      /^_+|_+$/g,
      ''
    );
}

function isMeaningfulLockValue(
  value
) {
  const cleaned =
    cleanText(value);

  if (!cleaned) {
    return false;
  }

  return ![
    'none',
    'n/a',
    'na',
    'null',
    'undefined',
    'unknown'
  ].includes(
    cleaned.toLowerCase()
  );
}

function normalizeAliases(
  value
) {
  if (Array.isArray(value)) {
    return value
      .map(
        item =>
          cleanText(item)
      )
      .filter(Boolean);
  }

  const cleaned =
    cleanText(value);

  if (
    !cleaned ||
    /^none$/i.test(cleaned)
  ) {
    return [];
  }

  return cleaned
    .split(',')
    .map(
      item =>
        cleanText(item)
    )
    .filter(Boolean);
}

function normalizeExpressionProfile(
  value
) {
  const source =
    value &&
    typeof value === 'object' &&
    !Array.isArray(value)
      ? value
      : {};

  return {
    baseline:
      cleanText(
        source.baseline
      ),

    focused:
      cleanText(
        source.focused ||
        source.pressionProfile
      ),

    concerned:
      cleanText(
        source.concerned
      ),

    inControl:
      cleanText(
        source.inControl
      ),

    confronting:
      cleanText(
        source.confronting
      ),

    challenged:
      cleanText(
        source.challenged
      ),

    exposed:
      cleanText(
        source.exposed
      ),

    defeated:
      cleanText(
        source.defeated
      ),

    relieved:
      cleanText(
        source.relieved
      )
  };
}

function normalizeCharacterLock(
  rawCharacter
) {
  if (
    !rawCharacter ||
    typeof rawCharacter !== 'object'
  ) {
    return null;
  }

  const name =
    cleanText(
      rawCharacter.name
    );

  if (!name) {
    return null;
  }

  const requestedId =
    cleanText(
      rawCharacter.id
    );

  const id =
    requestedId.startsWith('char_')
      ? requestedId
      : `char_${slugify(name)}`;

  const ageNumber =
    Number(
      rawCharacter.ageExact
    );

  const character = {
    id,

    name,

    aliases:
      normalizeAliases(
        rawCharacter.aliases
      ),

    narrativeRole:
      cleanText(
        rawCharacter.narrativeRole
      ),

    storyRole:
      cleanText(
        rawCharacter.storyRole
      ),

    ageExact:
      Number.isFinite(ageNumber) &&
      ageNumber > 0
        ? ageNumber
        : cleanText(
            rawCharacter.ageExact
          ),

    gender:
      cleanText(
        rawCharacter.gender
      ),

    skinTone:
      cleanText(
        rawCharacter.skinTone
      ),

    physique:
      cleanText(
        rawCharacter.physique
      ),

    faceShape:
      cleanText(
        rawCharacter.faceShape
      ),

    eyes:
      cleanText(
        rawCharacter.eyes
      ),

    nose:
      cleanText(
        rawCharacter.nose
      ),

    mouth:
      cleanText(
        rawCharacter.mouth
      ),

    jawline:
      cleanText(
        rawCharacter.jawline
      ),

    hairStyle:
      cleanText(
        rawCharacter.hairStyle
      ),

    hairColor:
      cleanText(
        rawCharacter.hairColor
      ),

    facialHair:
      cleanText(
        rawCharacter.facialHair
      ),

    clothing:
      cleanText(
        rawCharacter.clothing
      ),

    clothingColors:
      cleanText(
        rawCharacter.clothingColors
      ),

    footwear:
      cleanText(
        rawCharacter.footwear
      ),

    accessories:
      cleanText(
        rawCharacter.accessories
      ),

    distinguishingFeatures:
      cleanText(
        rawCharacter.distinguishingFeatures
      ),

    expressionProfile:
      normalizeExpressionProfile(
        rawCharacter.expressionProfile
      )
  };

  character.visualSignature =
    buildVisualSignature(
      character
    );

  return character;
}

function fieldFromBlock(
  block,
  field
) {
  const regex =
    new RegExp(
      `^${field}:\\s*(.*)$`,
      'im'
    );

  const match =
    block.match(
      regex
    );

  return cleanText(
    match?.[1]
  );
}

function parseLegacyCharacterLocks(
  text
) {
  const source =
    cleanText(text);

  if (!source) {
    return [];
  }

  const blocks =
    source
      .split(
        /\n(?=CHARACTER:\s*)/i
      )
      .map(
        item =>
          item.trim()
      )
      .filter(
        item =>
          /^CHARACTER:/i.test(
            item
          )
      );

  return blocks
    .map(
      block => {
        const name =
          fieldFromBlock(
            block,
            'CHARACTER'
          );

        if (!name) {
          return null;
        }

        return normalizeCharacterLock({
          name,

          aliases:
            fieldFromBlock(
              block,
              'ALIASES'
            ),

          narrativeRole:
            fieldFromBlock(
              block,
              'NARRATIVE ROLE'
            ),

          storyRole:
            fieldFromBlock(
              block,
              'STORY ROLE'
            ),

          ageExact:
            fieldFromBlock(
              block,
              'AGE EXACT'
            ),

          gender:
            fieldFromBlock(
              block,
              'GENDER'
            ),

          skinTone:
            fieldFromBlock(
              block,
              'SKIN TONE'
            ),

          physique:
            fieldFromBlock(
              block,
              'PHYSIQUE'
            ),

          faceShape:
            fieldFromBlock(
              block,
              'FACE SHAPE'
            ),

          eyes:
            fieldFromBlock(
              block,
              'EYES'
            ),

          nose:
            fieldFromBlock(
              block,
              'NOSE'
            ),

          mouth:
            fieldFromBlock(
              block,
              'MOUTH'
            ),

          jawline:
            fieldFromBlock(
              block,
              'JAWLINE'
            ),

          hairStyle:
            fieldFromBlock(
              block,
              'HAIR STYLE'
            ),

          hairColor:
            fieldFromBlock(
              block,
              'HAIR COLOR'
            ),

          facialHair:
            fieldFromBlock(
              block,
              'FACIAL HAIR'
            ),

          clothing:
            fieldFromBlock(
              block,
              'CLOTHING'
            ),

          clothingColors:
            fieldFromBlock(
              block,
              'CLOTHING COLORS'
            ),

          footwear:
            fieldFromBlock(
              block,
              'FOOTWEAR'
            ),

          accessories:
            fieldFromBlock(
              block,
              'ACCESSORIES'
            ),

          distinguishingFeatures:
            fieldFromBlock(
              block,
              'DISTINGUISHING FEATURES'
            ),

          expressionProfile: {}
        });
      }
    )
    .filter(Boolean);
}

function parseCharacterLocks(
  input
) {
  if (
    input === null ||
    input === undefined
  ) {
    return [];
  }

  let data =
    input;

  if (
    typeof input === 'string'
  ) {
    const source =
      cleanText(input);

    if (!source) {
      return [];
    }

    try {
      data =
        parseJsonOutput(
          source
        );
    } catch {
      return parseLegacyCharacterLocks(
        source
      );
    }
  }

  let rawCharacters = [];

  if (
    Array.isArray(data)
  ) {
    rawCharacters =
      data;
  } else if (
    data &&
    typeof data === 'object' &&
    Array.isArray(
      data.characters
    )
  ) {
    rawCharacters =
      data.characters;
  }

  const characters =
    rawCharacters
      .map(
        character =>
          normalizeCharacterLock(
            character
          )
      )
      .filter(Boolean);

  if (
    characters.length === 0
  ) {
    return [];
  }

  const usedIds =
    new Set();

  for (
    const character of
    characters
  ) {
    if (
      usedIds.has(
        character.id
      )
    ) {
      throw new Error(
        `DUPLICATE_CHARACTER_ID_${character.id}`
      );
    }

    usedIds.add(
      character.id
    );
  }

  return characters;
}

function buildVisualSignature(
  character
) {
  const pieces = [];

  const age =
    cleanText(
      character.ageExact
    );

  const gender =
    cleanText(
      character.gender
    );

  if (age && gender) {
    pieces.push(
      `${age}-year-old ${gender}`
    );
  } else if (age) {
    pieces.push(
      `${age}-year-old`
    );
  } else if (gender) {
    pieces.push(
      gender
    );
  }

  if (
    isMeaningfulLockValue(
      character.skinTone
    )
  ) {
    pieces.push(
      character.skinTone
    );
  }

  if (
    isMeaningfulLockValue(
      character.physique
    )
  ) {
    pieces.push(
      character.physique
    );
  }

  if (
    isMeaningfulLockValue(
      character.faceShape
    )
  ) {
    pieces.push(
      character.faceShape
    );
  }

  if (
    isMeaningfulLockValue(
      character.eyes
    )
  ) {
    pieces.push(
      character.eyes
    );
  }

  if (
    isMeaningfulLockValue(
      character.nose
    )
  ) {
    pieces.push(
      character.nose
    );
  }

  if (
    isMeaningfulLockValue(
      character.mouth
    )
  ) {
    pieces.push(
      character.mouth
    );
  }

  if (
    isMeaningfulLockValue(
      character.jawline
    )
  ) {
    pieces.push(
      character.jawline
    );
  }

  if (
    isMeaningfulLockValue(
      character.hairStyle
    )
  ) {
    pieces.push(
      character.hairStyle
    );
  }

  if (
    isMeaningfulLockValue(
      character.hairColor
    )
  ) {
    pieces.push(
      character.hairColor
    );
  }

  if (
    isMeaningfulLockValue(
      character.facialHair
    )
  ) {
    pieces.push(
      character.facialHair
    );
  }

  if (
    isMeaningfulLockValue(
      character.clothing
    )
  ) {
    pieces.push(
      `wearing ${character.clothing}`
    );
  }

  if (
    isMeaningfulLockValue(
      character.clothingColors
    )
  ) {
    pieces.push(
      `clothing colors ${character.clothingColors}`
    );
  }

  if (
    isMeaningfulLockValue(
      character.footwear
    )
  ) {
    pieces.push(
      `footwear ${character.footwear}`
    );
  }

  if (
    isMeaningfulLockValue(
      character.accessories
    )
  ) {
    pieces.push(
      `accessories ${character.accessories}`
    );
  }

  if (
    isMeaningfulLockValue(
      character.distinguishingFeatures
    )
  ) {
    pieces.push(
      `distinguishing features ${character.distinguishingFeatures}`
    );
  }

  return pieces.join(
    ', '
  );
}
/* =====================================================
   PREVIOUS OUTPUT LOOKUP
===================================================== */

function findPreviousByDestination(
  previousOutputs,
  destination
) {
  const target =
    cleanText(
      destination
    ).toUpperCase();

  for (
    let index =
      previousOutputs.length - 1;
    index >= 0;
    index -= 1
  ) {
    const item =
      previousOutputs[
        index
      ];

    if (
      item.outputDestination ===
      target
    ) {
      return item;
    }
  }

  return null;
}

function findSummaryOutput(
  previousOutputs
) {
  const general =
    previousOutputs.find(
      item =>
        item.outputDestination ===
        'GENERAL'
    );

  return general
    ? formatOutputForPrompt(
        general.output
      )
    : '';
}

/* =====================================================
   FINAL JSON — INTERNAL SCENE PLAN
===================================================== */

function buildCharacterCatalog(
  characters
) {
  return characters.map(
    character => ({
      id:
        character.id,

      name:
        character.name,

      narrativeRole:
        character.narrativeRole,

      storyRole:
        character.storyRole
    })
  );
}

function validateScenePlan(
  data
) {
  if (
    !data ||
    typeof data !==
      'object'
  ) {
    throw new Error(
      'SCENE_PLAN_INVALID'
    );
  }

  if (
    !Array.isArray(
      data.timeline
    ) ||
    data.timeline.length !==
      20
  ) {
    throw new Error(
      'TIMELINE_MUST_HAVE_20_ITEMS'
    );
  }

  if (
    !Array.isArray(
      data.scenes
    ) ||
    data.scenes.length !==
      20
  ) {
    throw new Error(
      'SCENES_MUST_HAVE_20_ITEMS'
    );
  }

  const expected =
    Array.from(
      { length: 20 },
      (_, index) =>
        index + 1
    );

  const actual =
    data.scenes.map(
      scene =>
        Number(
          scene?.sceneNumber
        )
    );

  const validNumbers =
    expected.every(
      (
        number,
        index
      ) =>
        actual[
          index
        ] === number
    );

  if (!validNumbers) {
    throw new Error(
      'SCENE_NUMBERS_MUST_BE_1_TO_20'
    );
  }

  return true;
}

function buildScenePlanPrompt({
  script,
  summary,
  instruction,
  characters
}) {
  const catalog =
    JSON.stringify(
      buildCharacterCatalog(
        characters
      ),
      null,
      2
    );

  return `
You are producing the INTERNAL SCENE PLAN for a story-production pipeline.

IMPORTANT:
The backend will add the final full image prompts later.
DO NOT generate imagePrompt text in this internal phase.

The user's workflow instruction remains authoritative for:
- story accuracy
- continuity
- exactly 20 scenes
- exactly 20 timeline events
- scene diversity
- story progression

CURRENT WORKFLOW INSTRUCTION:
${instruction}

LOCKED CHARACTER CATALOG:
Use ONLY these character IDs and names for recurring characters.
Do not redesign them.

${catalog}

STORY SUMMARY:
${summary}

TASK:
Read the FULL CURRENT SCRIPT and produce:

1. analysis
2. stable locations
3. stable props
4. stable vehicles
5. exactly 20 timeline events
6. exactly 20 scene plans

The 20 scenes must cover the complete story chronologically from opening through ending.

Avoid repetitive scenes.

Every scene must represent a distinct story beat.

For recurring locations, props, and vehicles:
create stable IDs and preserve identity and state.

RETURN VALID JSON ONLY.

Required structure:

{
  "analysis": {
    "title": "",
    "summary": "",
    "setting": "",
    "timePeriod": "",
    "mainConflict": "",
    "climax": "",
    "ending": ""
  },

  "locations": [
    {
      "id": "",
      "name": "",
      "type": "",
      "description": "",
      "keyFeatures": []
    }
  ],

  "props": [
    {
      "id": "",
      "name": "",
      "description": ""
    }
  ],

  "vehicles": [
    {
      "id": "",
      "name": "",
      "description": ""
    }
  ],

  "timeline": [
    {
      "sceneNumber": 1,
      "title": "",
      "storyMoment": "",
      "purpose": "",
      "characterIds": [],
      "locationId": "",
      "propIds": [],
      "vehicleIds": [],
      "timeOfDay": "",
      "event": "",
      "continuityState": ""
    }
  ],

  "scenes": [
    {
      "sceneNumber": 1,
      "title": "",
      "storyMoment": "",
      "purpose": "",
      "characterIds": [],
      "locationId": "",
      "propIds": [],
      "vehicleIds": [],
      "timeOfDay": "",
      "action": "",
      "emotion": "",
      "cameraSuggestion": "",
      "foreground": "",
      "midground": "",
      "background": "",
      "lighting": "",
      "continuity": ""
    }
  ]
}

STRICT VALIDATION:
- timeline length must equal 20
- scenes length must equal 20
- sceneNumber must run exactly 1 through 20
- use only locked recurring character IDs
- do not create imagePrompt
- no markdown fences
- no text before or after JSON

==================================================
FULL CURRENT SCRIPT
==================================================

${script}
`;
}

async function generateScenePlan(
  args
) {
  let lastError = null;

  for (
    let attempt = 1;
    attempt <= 2;
    attempt += 1
  ) {
    try {
      const prompt =
        buildScenePlanPrompt(
          args
        ) +
        (
          attempt === 2
            ? `

IMPORTANT RETRY:
The previous attempt failed structural validation.
Return one COMPLETE valid JSON object.
Exactly 20 timeline items.
Exactly 20 scene items.
Do not truncate the response.
`
            : ''
        );

      const raw =
        await runGeminiStep(
          prompt,
          0.2
        );

      const parsed =
        parseJsonOutput(
          raw
        );

      validateScenePlan(
        parsed
      );

      return parsed;

    } catch (error) {
      lastError =
        error;
    }
  }

  throw lastError ||
    new Error(
      'SCENE_PLAN_FAILED'
    );
}

/* =====================================================
   IMAGE PROMPT COMPOSER
===================================================== */

function characterMapFromList(
  characters
) {
  const map =
    new Map();

  for (
    const character of
    characters
  ) {
    map.set(
      character.id,
      character
    );
  }

  return map;
}

function buildSceneImagePrompt(
  scene,
  characterMap
) {
  const characterIds =
    Array.isArray(
      scene.characterIds
    )
      ? scene.characterIds
      : [];

  const visibleCharacters =
    characterIds
      .map(
        id =>
          characterMap.get(
            id
          )
      )
      .filter(Boolean);

  const identityText =
    visibleCharacters.length
      ? visibleCharacters
          .map(
            character =>
              `${character.name}: ${buildVisualSignature(character)}`
          )
          .join(
            '. '
          )
      : 'No recurring locked character visible in this scene';

  const parts = [
    `LOCKED CHARACTER IDENTITIES: ${identityText}.`,

    scene.action
      ? `ACTION: ${scene.action}.`
      : '',

    scene.emotion
      ? `EXPRESSION AND EMOTION: ${scene.emotion}.`
      : '',

    scene.foreground
      ? `FOREGROUND: ${scene.foreground}.`
      : '',

    scene.midground
      ? `MIDGROUND: ${scene.midground}.`
      : '',

    scene.background
      ? `BACKGROUND: ${scene.background}.`
      : '',

    scene.cameraSuggestion
      ? `CAMERA: ${scene.cameraSuggestion}.`
      : '',

    scene.timeOfDay
      ? `TIME: ${scene.timeOfDay}.`
      : '',

    scene.lighting
      ? `LIGHTING: ${scene.lighting}.`
      : '',

    scene.continuity
      ? `CONTINUITY: ${scene.continuity}.`
      : '',

    'The same locked recurring character must remain visually identical across all scenes. Do not change age, face, eyes, hair, facial hair, skin tone, physique, clothing type, clothing colors, or distinguishing features.',

    'Ultra photorealistic live-action professional DSLR photograph, realistic anatomy, natural physically accurate lighting, authentic skin and fabric texture, natural shadows, true-to-life colors, tack-sharp primary faces and eyes, realistic environment, no CGI, no illustration, no cartoon, no plastic skin.'
  ];

  return parts
    .filter(Boolean)
    .join(' ');
}

/* =====================================================
   FINAL JSON PIPELINE
===================================================== */

async function runFinalJsonPipeline({
  script,
  instruction,
  previousOutputs
}) {
  const characterStep =
    findPreviousByDestination(
      previousOutputs,
      'CHARACTER_LOCKS'
    );

  if (!characterStep) {
    throw new Error(
      'CHARACTER_LOCKS_REQUIRED_BEFORE_FINAL_JSON'
    );
  }

  const characterText =
    formatOutputForPrompt(
      characterStep.output
    );

  const characters =
    parseCharacterLocks(
      characterText
    );

  if (
    characters.length === 0
  ) {
    throw new Error(
      'CHARACTER_LOCK_PARSE_FAILED'
    );
  }

  const summary =
    findSummaryOutput(
      previousOutputs
    );

  const scenePlan =
    await generateScenePlan({
      script,
      summary,
      instruction,
      characters
    });

  const characterMap =
    characterMapFromList(
      characters
    );

  const scenes =
    scenePlan.scenes.map(
      scene => ({
        sceneNumber:
          scene.sceneNumber,

        title:
          cleanText(
            scene.title
          ),

        storyMoment:
          cleanText(
            scene.storyMoment
          ),

        purpose:
          cleanText(
            scene.purpose
          ),

        characterIds:
          Array.isArray(
            scene.characterIds
          )
            ? scene.characterIds
            : [],

        locationId:
          cleanText(
            scene.locationId
          ),

        propIds:
          Array.isArray(
            scene.propIds
          )
            ? scene.propIds
            : [],

        vehicleIds:
          Array.isArray(
            scene.vehicleIds
          )
            ? scene.vehicleIds
            : [],

        timeOfDay:
          cleanText(
            scene.timeOfDay
          ),

        action:
          cleanText(
            scene.action
          ),

        emotion:
          cleanText(
            scene.emotion
          ),

        cameraSuggestion:
          cleanText(
            scene.cameraSuggestion
          ),

        continuity:
          cleanText(
            scene.continuity
          ),

        imagePrompt:
          buildSceneImagePrompt(
            scene,
            characterMap
          )
      })
    );

  if (
    scenes.length !==
    20
  ) {
    throw new Error(
      'FINAL_SCENES_MUST_HAVE_20_ITEMS'
    );
  }

  return {
    analysis:
      scenePlan.analysis ||
      {},

    characters,

    locations:
      Array.isArray(
        scenePlan.locations
      )
        ? scenePlan.locations
        : [],

    props:
      Array.isArray(
        scenePlan.props
      )
        ? scenePlan.props
        : [],

    vehicles:
      Array.isArray(
        scenePlan.vehicles
      )
        ? scenePlan.vehicles
        : [],

    timeline:
      scenePlan.timeline,

    scenes
  };
}

/* =====================================================
   VIDEO HOOK PIPELINE
===================================================== */

function buildHookPlanPrompt({
  script,
  summary,
  instruction,
  characters
}) {
  const catalog =
    JSON.stringify(
      buildCharacterCatalog(
        characters
      ),
      null,
      2
    );

  return `
You are selecting exactly 5 distinct VIDEO HOOK MOMENTS.

The backend will automatically inject the permanent Character Locks into every final hook.

Do NOT redesign characters.
Do NOT write full character appearance descriptions yourself.

CURRENT USER INSTRUCTION:
${instruction}

LOCKED CHARACTER CATALOG:
${catalog}

STORY SUMMARY:
${summary}

TASK:
Select exactly 5 visually strong and clearly different hook moments from the current script.

The hooks must:
- be based on real story events
- not all use the same composition
- not all be close-up arguments
- avoid unnecessary ending spoilers
- represent different forms of tension, discovery, accusation, evidence, confrontation, or emotional reaction where appropriate
- use only valid locked recurring character IDs

Return valid JSON only as an array with exactly 5 objects:

[
  {
    "hookNumber": 1,
    "characterIds": [],
    "storyMoment": "",
    "action": "",
    "expression": "",
    "environment": "",
    "camera": "",
    "lighting": ""
  }
]

No markdown fences.
No commentary.

==================================================
FULL CURRENT SCRIPT
==================================================

${script}
`;
}

async function generateHookPlan(
  args
) {
  let lastError =
    null;

  for (
    let attempt = 1;
    attempt <= 2;
    attempt += 1
  ) {
    try {
      const prompt =
        buildHookPlanPrompt(
          args
        );

      const raw =
        await runGeminiStep(
          prompt,
          0.4
        );

      const parsed =
        parseJsonOutput(
          raw
        );

      if (
        !Array.isArray(
          parsed
        ) ||
        parsed.length !== 5
      ) {
        throw new Error(
          'HOOK_PLAN_MUST_HAVE_5_ITEMS'
        );
      }

      return parsed;

    } catch (error) {
      lastError =
        error;
    }
  }

  throw lastError ||
    new Error(
      'HOOK_PLAN_FAILED'
    );
}

function composeHookPrompt(
  hook,
  characterMap
) {
  const ids =
    Array.isArray(
      hook.characterIds
    )
      ? hook.characterIds
      : [];

  const visibleCharacters =
    ids
      .map(
        id =>
          characterMap.get(
            id
          )
      )
      .filter(Boolean);

  const identityText =
    visibleCharacters.length
      ? visibleCharacters
          .map(
            character =>
              `${character.name}: ${buildVisualSignature(character)}`
          )
          .join(
            '. '
          )
      : 'No recurring locked character visible';

  const parts = [
    `LOCKED CHARACTER IDENTITIES: ${identityText}.`,

    hook.action
      ? `ACTION: ${hook.action}.`
      : '',

    hook.expression
      ? `EXPRESSION: ${hook.expression}.`
      : '',

    hook.environment
      ? `ENVIRONMENT: ${hook.environment}.`
      : '',

    hook.camera
      ? `CAMERA: ${hook.camera}.`
      : '',

    hook.lighting
      ? `LIGHTING: ${hook.lighting}.`
      : '',

    'Maintain the exact same locked face, age, skin tone, hair, facial hair, physique, wardrobe, clothing colors, and distinguishing features.',

    'Photorealistic live-action cinematic video, realistic anatomy, realistic skin and fabric texture, natural physically accurate lighting, authentic environment, true-to-life colors, no CGI look, no cartoon look, no plastic skin.'
  ];

  return parts
    .filter(Boolean)
    .join(' ');
}

async function runVideoHooksPipeline({
  script,
  instruction,
  previousOutputs
}) {
  const characterStep =
    findPreviousByDestination(
      previousOutputs,
      'CHARACTER_LOCKS'
    );

  if (!characterStep) {
    throw new Error(
      'CHARACTER_LOCKS_REQUIRED_BEFORE_VIDEO_HOOKS'
    );
  }

  const characters =
    parseCharacterLocks(
      formatOutputForPrompt(
        characterStep.output
      )
    );

  if (
    characters.length === 0
  ) {
    throw new Error(
      'CHARACTER_LOCK_PARSE_FAILED'
    );
  }

  const summary =
    findSummaryOutput(
      previousOutputs
    );

  const hookPlan =
    await generateHookPlan({
      script,
      summary,
      instruction,
      characters
    });

  const characterMap =
    characterMapFromList(
      characters
    );

  return hookPlan
    .map(
      (
        hook,
        index
      ) =>
        `HOOK ${index + 1}: ${composeHookPrompt(
          hook,
          characterMap
        )}`
    )
    .join(
      '\n\n'
    );
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
        'SEQUENTIAL_DYNAMIC_WORKFLOW_V2_LOCKED_CHARACTER_COMPOSER',

      model:
        MODEL_NAME,

      geminiConfigured:
        Boolean(
          API_KEY
        )
    });
  }
);

/* =====================================================
   RUN STEP
===================================================== */

app.post(
  '/run-step',
  async (
    req,
    res
  ) => {
    try {
      const script =
        cleanText(
          req.body?.script
        );

      const stepName =
        cleanText(
          req.body?.stepName
        ) ||
        'Workflow Step';

      const instruction =
        cleanText(
          req.body?.instruction
        );

      const outputType =
        cleanText(
          req.body?.outputType
        ).toUpperCase() ===
        'JSON'
          ? 'JSON'
          : 'TEXT';

      const outputDestination =
        cleanText(
          req.body?.outputDestination
        ).toUpperCase() ||
        'GENERAL';

      const previousOutputs =
        normalizePreviousOutputs(
          req.body?.previousOutputs
        );

      if (!script) {
        return res
          .status(400)
          .json({
            ok: false,
            error:
              'SCRIPT_REQUIRED'
          });
      }

      if (!instruction) {
        return res
          .status(400)
          .json({
            ok: false,
            error:
              'STEP_INSTRUCTION_REQUIRED'
          });
      }

      /* =============================================
         SPECIAL FINAL JSON PIPELINE
      ============================================= */

      if (
        outputDestination ===
        'FINAL_JSON'
      ) {
        const output =
          await runFinalJsonPipeline({
            script,
            instruction,
            previousOutputs
          });

        return res.json({
          ok: true,
          stepName,
          outputType:
            'JSON',
          outputDestination,
          output
        });
      }

      /* =============================================
         SPECIAL VIDEO HOOK PIPELINE
      ============================================= */

      if (
        outputDestination ===
        'VIDEO_HOOKS'
      ) {
        const output =
          await runVideoHooksPipeline({
            script,
            instruction,
            previousOutputs
          });

        return res.json({
          ok: true,
          stepName,
          outputType:
            'TEXT',
          outputDestination,
          output
        });
      }

      /* =============================================
         STANDARD STEP
      ============================================= */

      const prompt =
        buildStepPrompt({
          script,
          stepName,
          instruction,
          previousOutputs,
          outputType
        });

      const rawOutput =
        await runGeminiStep(
          prompt,
          0.4
        );

      const output =
        outputType ===
        'JSON'
          ? parseJsonOutput(
              rawOutput
            )
          : rawOutput;

      return res.json({
        ok: true,
        stepName,
        outputType,
        outputDestination,
        output
      });

    } catch (error) {
      console.error(
        '[RUN_STEP_FAILED]',
        error
      );

      const message =
        error?.message ||
        'RUN_STEP_FAILED';

      if (
        message.includes(
          '429'
        ) ||
        message
          .toLowerCase()
          .includes(
            'quota'
          )
      ) {
        return res
          .status(429)
          .json({
            ok: false,
            error:
              'GEMINI_USAGE_LIMIT',
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

      return res
        .status(500)
        .json({
          ok: false,
          error:
            message,
          detail:
            error?.detail ||
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
      `Story Pipeline Engine running on port ${PORT}`
    );

    console.log(
      `Model: ${MODEL_NAME}`
    );

    console.log(
      `Mode: SEQUENTIAL_DYNAMIC_WORKFLOW_V2_LOCKED_CHARACTER_COMPOSER`
    );
  }
);
