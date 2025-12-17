


def study_agent_msg():
    # Define system message
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType

    sys_msg = "You are a helpful assistant."

    # Set agent
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=ModelType.DEEPSEEK_CHAT,
    )
    camel_agent = ChatAgent(system_message=sys_msg, model=model)

    # Set user message
    user_msg = """Say hi to CAMEL AI, one open-source community dedicated to the
        study of autonomous and communicative agents."""

    # Get response information
    response = camel_agent.step(user_msg)
    print(response.msgs[0].content)

def study_base_message():
    from io import BytesIO

    import requests
    from PIL import Image

    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType

    # URL of the image
    url = "https://raw.githubusercontent.com/camel-ai/camel/master/misc/logo_light.png"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Define system message
    sys_msg = BaseMessage.make_assistant_message(
        role_name="Assistant",
        content="You are a helpful assistant.",
    )

    # Set agent
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_3_PRO,
    )
    camel_agent = ChatAgent(system_message=sys_msg, model=model)

    # Set user message
    user_msg = BaseMessage.make_user_message(
        role_name="User", content="""what's in the image?""", image_list=[img]
    )

    # Get response information
    response = camel_agent.step(user_msg)
    print(response.msgs[0].content)

def study_camel_prompt():
    from camel.agents import TaskSpecifyAgent
    from camel.configs import ChatGPTConfig
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType, TaskType

    # Set up the model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_3_PRO,
    )

    # Create a task specify agent
    task_specify_agent = TaskSpecifyAgent(
        model=model, task_type=TaskType.AI_SOCIETY
    )

    # Run the agent with a task prompt
    specified_task_prompt = task_specify_agent.run(
        task_prompt="Improving stage presence and performance skills",
        meta_dict=dict(
            assistant_role="Musician", user_role="Student", word_limit=100
        ),
    )

    print(f"Specified task prompt:\n{specified_task_prompt}\n")

def study_codePromptTemplateDict():
    from camel.prompts import CodePromptTemplateDict

    # Generate programming languages
    languages_prompt = CodePromptTemplateDict.GENERATE_LANGUAGES.format(num_languages=5)
    print(f"Languages prompt:\n{languages_prompt}\n")

    # Generate coding tasks
    tasks_prompt = CodePromptTemplateDict.GENERATE_TASKS.format(num_tasks=3)
    print(f"Tasks prompt:\n{tasks_prompt}\n")

    # Create an AI coding assistant prompt
    assistant_prompt = CodePromptTemplateDict.ASSISTANT_PROMPT.format(
        assistant_role="Python Expert",
        task_description="Implement a binary search algorithm",
    )
    print(f"Assistant prompt:\n{assistant_prompt}\n")

    from camel.prompts import TranslationPromptTemplateDict

    translation_prompt = TranslationPromptTemplateDict.ASSISTANT_PROMPT.format(target_language="Spanish")
    print(f"Translation prompt:\n{translation_prompt}\n")

    from camel.prompts import EvaluationPromptTemplateDict

# Generate evaluation questions
    questions_prompt = EvaluationPromptTemplateDict.GENERATE_QUESTIONS.format(
        num_questions=5,
        field="Machine Learning",
        examples="1. What is the difference between supervised and unsupervised learning?\n2. Explain the concept of overfitting.",
    )
    print(f"Evaluation questions prompt:\n{questions_prompt}\n")



    from camel.prompts import ObjectRecognitionPromptTemplateDict

    # Create an object recognition assistant prompt
    recognition_prompt = ObjectRecognitionPromptTemplateDict.ASSISTANT_PROMPT
    print(f"Object recognition prompt:\n{recognition_prompt}\n")

def study_tools():
    # Import the necessary tools
    from camel.toolkits import MathToolkit, SearchToolkit
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType, TaskType

    sys_msg = 'You are a curious stone wondering about the universe.'
    model = ModelFactory.create(
    model_platform=ModelPlatformType.GEMINI,
    model_type=ModelType.GEMINI_3_PRO,
    )


    # Initialize the agent with list of tools
    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools = [
            *MathToolkit().get_tools(),
            *SearchToolkit().get_tools(),
        ]
        )
    # agent.set_output_language('chinese')
    agent.output_language = 'chinese'
    # Let agent step the message
    response = agent.step("What is CAMEL AI?")

    # Check tool calling
    print(response.info['tool_calls'])
    print(agent.memory.get_context())
    # Get response content
    print(response.msgs[0].content)

def agent_society():
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType, TaskType
    from camel.societies import RolePlaying

    model = ModelFactory.create(
    model_platform=ModelPlatformType.GEMINI,
    model_type=ModelType.GEMINI_3_PRO,
    )

    def is_terminated(response):
        """
        Give alerts when the session should be terminated.
        """
        if response.terminated:
            role = response.msg.role_type.name
            reason = response.info['termination_reasons']
            print(f'AI {role} terminated due to {reason}')

        return response.terminated

    def run(society, round_limit: int=10):

        # Get the initial message from the ai assistant to the ai user
        input_msg = society.init_chat()

        # Starting the interactive session
        for _ in range(round_limit):

            # Get the both responses for this round
            assistant_response, user_response = society.step(input_msg)

            # Check the termination condition
            if is_terminated(assistant_response) or is_terminated(user_response):
                break

            # Get the results
            print(f'[AI User] {user_response.msg.content}.\n')
            # Check if the task is end
            if 'CAMEL_TASK_DONE' in user_response.msg.content:
                break
            print(f'[AI Assistant] {assistant_response.msg.content}.\n')



            # Get the input message for the next round
            input_msg = assistant_response.msg

        return None
    task_kwargs = {
        'task_prompt': 'Develop a plan to TRAVEL TO THE PAST and make changes.',
        'with_task_specify': True,
        'task_specify_agent_kwargs': {'model': model}
    }

    user_role_kwargs = {
    'user_role_name': 'an ambitious aspiring TIME TRAVELER',
    'user_agent_kwargs': {'model': model}
    }

    assistant_role_kwargs = {
    'assistant_role_name': 'the best-ever experimental physicist',
    'assistant_agent_kwargs': {'model': model}
    }

    society = RolePlaying(
    **task_kwargs,             # The task arguments
    **user_role_kwargs,        # The instruction sender's arguments
    **assistant_role_kwargs,   # The instruction receiver's arguments
    output_language="chinese"
)

    run(society)
    # Create a translation assistant prompt
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    # study_agent_msg()

    # study_camel_prompt()
    # study_codePromptTemplateDict()
    # study_tools()
    agent_society()
  