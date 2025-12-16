


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

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    # study_agent_msg()

    study_base_message()
  