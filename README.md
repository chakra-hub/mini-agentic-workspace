




Mini Agentic Workspace
This is a small project I built to learn and demonstrate how multiple AI agents can work together on a software task.

The idea is simple: a user gives the application a task. The application estimates how difficult the task is, chooses models for the different agents, and then runs the work through a few stages:

Plan the work.
Research the technical topic.
Implement the solution.
Review the implementation.
Send it back to the Developer if changes are needed.
I used LangGraph to control the workflow, Deep Agents to create the specialist agents, and OpenRouter to access the models.

Current status: The mock version of the workflow and its automated tests are working. The live OpenRouter classifier and the complete live multi-agent workflow still need to be tested successfully. During development, the OpenRouter free-model daily limit was exhausted, so I could not finish those live tests.

Why I built this
I wanted to build a small example of an agentic development workflow rather than just create one chatbot that answers a question.


What the project does
The application takes a task and runs it through a LangGraph workflow.

1. It classifies the task
The Smart Router estimates whether the task is simple, medium or complex. In live mode, it asks a classifier model through OpenRouter and validates the response using Pydantic.

The classifier looks at more than the length of the request. It considers things such as:

How much reasoning is needed.
Whether the task needs research.
Whether it should be split into smaller tasks.
Whether external tools or files are needed.
How confident the classifier is in its decision.
2. It chooses a model for each agent
The application does not automatically give every agent the same model. It uses a small role-based policy to decide which model tier each agent should use.

For example, for a complex project, the Researcher and Reviewer use the medium model, while the Developer uses the complex model. The thinking behind this is that the implementation stage may need a stronger model, while the other stages may not.

This is a starting policy for the POC. It still needs to be compared against real task quality, response time and model cost before I can say it is the best allocation.

3. It runs three specialist agents
The workflow has three specialist agents:

Researcher: looks into the technical topic and prepares a report.
Developer: uses the requirements and research report to prepare an implementation and tests.
Reviewer: checks the implementation and gives feedback.
Each agent has its own instructions and its own SKILL.md file.

4. It supports a review and revision cycle
The Reviewer can return either:

APPROVED
CHANGES_REQUESTED
When changes are requested, the Developer receives the review feedback and gets another chance to revise the implementation. The workflow has a revision limit so it cannot keep looping forever.

The current demo allows one revision.

5. It records usage
The application keeps a usage record for the classifier and each specialist-agent call. Calls made during a revision are recorded too.

When model responses include token usage, the application records input, output and total tokens. It does not guess the actual billed cost when that information is not available.

How the workflow is connected
This is the current flow:

User task
   |
   v
Smart Router
   |
   v
Planner
   |
   v
Researcher
   |
   v
Developer
   |
   v
Reviewer
   |
   +---- Approved --------------------------> End
   |
   +---- Changes requested, revision left -> Developer
   |
   +---- Revision limit reached -----------> End with findings
LangGraph controls this sequence. The agents focus on their own jobs; they do not decide the overall order of the application.

I chose this approach because it makes the workflow easier to understand, test and change. For example, the review decision and revision limit are normal graph logic, not something I leave entirely up to a model.

The agents
Planner
The Planner is currently written as normal Python logic. It creates an execution plan from the user's task and lists the main stages and acceptance criteria.

It does not call an LLM. I kept it deterministic because planning the basic workflow does not always need another model call.

Researcher
The Researcher prepares a technical report. Its instructions ask it to explain important concepts, compare options, describe trade-offs and include production considerations where useful.

It has access to:

skills/research/SKILL.md

Developer
The Developer receives the original task and the Researcher's report. It is asked to produce maintainable Python code, type hints, input validation, error handling and relevant pytest tests.

When a review asks for changes, the Developer also receives the previous implementation and the review feedback.

It has access to:

skills/software-development/SKILL.md

Reviewer
The Reviewer checks the implementation against the original requirements. Its instructions cover correctness, edge cases, validation, security, maintainability and test quality.

It has access to:

skills/code-review/SKILL.md

The live Reviewer is asked to finish with an explicit status marker. If the marker is missing or invalid, the workflow treats the result as requiring changes rather than approving it automatically.

How the skills work
I keep the agent-specific instructions in separate SKILL.md files instead of putting every instruction into one large Python prompt.

The project currently has three skills:

skills/
├── research/
│   └── SKILL.md
├── software-development/
│   └── SKILL.md
└── code-review/
    └── SKILL.md
The skill loader searches for skills/*/SKILL.md, reads each file and converts it into file data for the Deep Agents StateBackend.

The agents are configured with the /skills/ directory as their skills source. In a live run, the agent can discover the skill descriptions and read the relevant instructions when needed.

There is an important difference between loading a skill and proving that an agent used it. The automated tests currently check that the files are found and loaded. I still need a successful live run and trace to confirm that each agent actually reads and follows its skill.

Smart model routing
The project has three model tiers. The model IDs are stored in .env, so they can be changed without editing the workflow code.

Tier	Intended use
Simple	Short explanations and straightforward tasks
Medium	Moderate reasoning, research and regular coding tasks
Complex	Larger implementation tasks and architecture work
The live classifier returns structured information, including a score from 0 to 100, task type, confidence, and flags for research, tools and decomposition. Pydantic checks the response before the routing policy uses it.

The current score bands are:

Score	Tier
0–34	Simple
35–69	Medium
70–100	Complex
The policy can also move a task to a higher tier if it needs decomposition or tools. A low-confidence result can also be moved up one tier.

Current agent model policy
Overall task complexity	Researcher	Developer	Reviewer
Simple	Simple	Simple	Simple
Medium	Medium	Medium	Medium
Complex	Medium	Complex	Medium
For example, the complex RAG architecture task was assigned a score of 85 in mock mode. The mock router selected the medium model for Researcher and Reviewer and the complex model for Developer.

The simple, medium and complex examples have all been tested with the deterministic mock rules. They have not been used to measure the accuracy of the live LLM classifier.

Mock mode and live mode
The project has two execution modes. Set the mode in .env.

Mock mode
EXECUTION_MODE=mock
Mock mode does not call an LLM API. It is useful for working on the graph, checking routing rules, testing the revision path and running the automated tests without using API quota.

The mock classifier uses simple rules. The Researcher, Developer and Reviewer return prepared example outputs. These outputs are labelled as simulated.

There are a few limitations to keep in mind:

The mock classifier is not the live LLM classifier.
The prepared Researcher and Developer outputs are examples for the RRF task. They do not change to match every task a user enters.
The mock Developer's revision is a demonstration. It does not regenerate the code.
The mock Reviewer follows a scripted result. It does not run tests or prove that the implementation is correct.
So mock mode proves that the application flow works. It does not prove that the agents can complete arbitrary tasks.

Live mode
EXECUTION_MODE=live
Live mode is intended to call the classifier and then invoke the selected Deep Agents through OpenRouter.

It needs a valid API key and model IDs in .env. It also depends on the model being available and the OpenRouter account having enough quota.

I have integrated the live code path, but I have not yet verified the complete live workflow from classification through review. The live classifier test returned an HTTP 429 error because the free-model daily quota had been used up. I will need to run the live checks again when quota is available.

Usage and cost tracking
The usage history records calls for:

model_router_classifier
researcher
developer
reviewer
Each invocation gets its own record. This includes a second Developer or Reviewer invocation when the workflow goes through a revision.

In mock mode, records are marked as mock and token counts are left unavailable. The summary reports zero API cost because no LLM API calls were made.

In live mode, the application records token usage when it is included in the model response. The actual USD cost calculation is not finished yet. Token counts alone are not enough to claim an exact bill, so the application currently leaves live cost unknown rather than making up a number.

A next improvement is to add model pricing information and calculate an estimated cost from input and output tokens. That estimate should be labelled separately from provider-reported billed cost.

Project structure
The code is split into folders so that each part has one main job.

mini-agentic-workspace/
│
├── app/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── state.py
│   │
│   ├── agents/
│   │   ├── factory.py
│   │   └── prompts.py
│   │
│   ├── model_router/
│   │   ├── classifier.py
│   │   └── policy.py
│   │
│   ├── skills/
│   │   └── loader.py
│   │
│   ├── workflow/
│   │   ├── graph.py
│   │   └── nodes/
│   │       ├── router.py
│   │       ├── planner.py
│   │       ├── researcher.py
│   │       ├── developer.py
│   │       └── reviewer.py
│   │
│   ├── observability/
│   │   └── usage.py
│   │
│   ├── demo/
│   │   └── mock_outputs.py
│   │
│   └── utils/
│       └── messages.py
│
├── skills/
│   ├── research/SKILL.md
│   ├── software-development/SKILL.md
│   └── code-review/SKILL.md
│
├── tests/
│   ├── test_model_router.py
│   ├── test_skills.py
│   └── test_workflow.py
│
├── backup/                 # Local copies of older files
├── docs/                   # Additional project notes
├── scripts/                # Helper scripts
│
├── .env                    # Local settings and API key; do not commit
├── .gitignore
├── requirements.txt
├── README.md
└── agent_workflow.py       # Compatibility launcher
The main way to start the application is python -m app. The agent_workflow.py file is kept as a small compatibility launcher.

Setup
I tested the project in a Python 3.14 virtual environment on macOS.

1. Get the project
git clone <your-repository-url>
cd mini-agentic-workspace
2. Create a virtual environment
macOS or Linux:

python -m venv .venv
source .venv/bin/activate
Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Add environment settings
Create a .env file in the project root. For mock mode, set EXECUTION_MODE=mock. You do not need an OpenRouter key to run the mock workflow.

For live mode, add your OpenRouter API key and the model IDs you want to use.

Configuration
The project reads these values from .env:

# Choose mock or live
EXECUTION_MODE=mock

# Needed for live mode
OPENROUTER_API_KEY=

# Classifier model
OPENROUTER_CLASSIFIER_MODEL=baidu/ernie-5.5-flash-preview:free

# Model tiers
OPENROUTER_SIMPLE_MODEL=liquid/lfm-3-1.7b:free
OPENROUTER_MEDIUM_MODEL=baidu/ernie-5.5-flash-preview:free
OPENROUTER_COMPLEX_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
These are the model IDs used during development. Model availability and free-tier limits can change, so check your OpenRouter account before relying on a particular model.

Do not commit .env or share your API key. An .env.example file should be added before publishing the repository; it should contain the variable names and an empty API key value, not a real secret.

Run the application
From the project root:

python -m app
You can also use the compatibility launcher:

python agent_workflow.py
The current CLI runs the example task configured in app/cli.py. To try a different task, change the task in that file.

Remember that changing the task in mock mode changes the routing input, but the prepared agent outputs are still RRF examples. Use live mode for task-specific model responses once the live path has been verified and quota is available.

Run the tests
Run the full test suite:

python -m pytest tests/ -v
Run only the model-router tests:

python -m pytest tests/test_model_router.py -v
Check that the Python files compile:

python -m compileall -q app
Tests currently in the project
The current test suite has 19 passing tests:

Model router: simple, medium and complex mock classification; model-tier selection; invalid complexity; empty task; low-confidence escalation; tool-based escalation.
Skills: finding skill files, reading their content, and handling a missing or empty skills directory.
Workflow: graph construction, approval, change requests, revision limits and default review behaviour.
These tests run locally without using OpenRouter quota. They test the application's deterministic logic, not the quality of live model responses.

What I have tested so far
Here is what I have actually run during development:

Check	Result
Python compilation of the modular application	Passed
Start application with python -m app in mock mode	Passed
Simple mock routing example	Passed; score 20
Medium mock routing example	Passed; score 52
Complex mock routing example	Passed; score 85
Role-based model selection	Passed in mock mode
LangGraph graph construction	Passed
Mock review-feedback loop with one revision	Passed
Skill-file discovery and loading tests	Passed
Automated test suite	19 passed
Live OpenRouter classifier	Not completed; free daily quota returned HTTP 429
Full live Deep Agent workflow	Not yet verified
What I learned while building this
Model routing and provider routing are two different things
My application decides which model ID to use based on the task and the agent's job. OpenRouter is the API gateway I use to access the selected model. The application choosing a model and the provider handling the request are separate parts of the process.

The strongest model is not always needed
A complex project does not mean every agent has to use the complex model. The Researcher may be able to prepare a useful report with a medium model, while the Developer may need the stronger model for implementation. This is why I added a separate model choice for each agent role.

LangGraph is useful for controlling the workflow
The graph makes the order of work clear. It also keeps the review decision and retry limit outside the LLM. That makes the process easier to inspect and test.

Deep Agents and skills help keep agent instructions organised
Each specialist has its own role and can use a separate skill file. This keeps reusable instructions out of one large prompt and makes them easier to update.

Model output needs validation
A model can return incomplete or badly formatted output. The classifier response is checked with Pydantic before the application uses it for routing. The Reviewer also needs to return a clear decision marker before the workflow can approve the work.

Review loops need a stopping point
Sending code back and forth between a Developer and Reviewer can become an endless loop. The revision counter gives the workflow a clear limit and makes the behaviour testable.

Mock mode is useful, but it is not live agent execution
Mock mode helped me build the graph and test the routing logic without making API calls. It also made it clear that a successful mock run does not prove that an LLM generated the code, used the skill or approved the implementation correctly. Those things need a live test.

Token usage and API cost are not the same
Token counts are useful, but they do not automatically tell me the exact amount billed. I need the model's pricing and reliable usage data to estimate cost. I have not finished that part yet, so the application does not claim an exact live cost.

Splitting code into modules makes it easier to work on
I started with a much larger workflow file. I then separated configuration, model routing, agent prompts, skill loading, workflow nodes and usage tracking. Now I can work on one part without changing the whole application each time.

AIDLC is more than running several agents
The important part is how the work moves between stages. Planning, passing research to development, reviewing the result, sending feedback back and deciding when to stop are all part of the lifecycle.

Things I still want to improve
This is a proof of concept, so there are a few things I still want to finish:

Run the live classifier once OpenRouter quota is available.
Run each Deep Agent with its selected model and check the actual response.
Check through live traces that each agent reads and follows its SKILL.md instructions.
Run the full live workflow, including a real Developer revision after Reviewer feedback.
Extract generated tests and run them in a controlled environment before treating a review as approved.
Add model pricing and report estimated cost per agent and per workflow. Keep estimated cost separate from the actual billed amount.
Improve the Planner so it can create task-specific steps and acceptance criteria instead of using a fixed template.
Make mock examples clearer and provide different fixtures for different scenarios.
Add .env.example and finish the repository setup files before publishing.
Final note
I built this project to get hands-on experience with model routing, LangGraph, Deep Agents, reusable skills and a development workflow with review feedback.

The local mock workflow and the 19 automated tests are working. The live OpenRouter and full live multi-agent runs are still to be verified. I have kept that distinction clear so the project shows what is working today and what still needs to be tested.





