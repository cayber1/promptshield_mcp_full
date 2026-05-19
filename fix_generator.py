content = open('agents/generator_agent.py').read()
content = content.replace(
    'client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))',
    'client = openai.OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")'
)
content = content.replace('model=model,', 'model="llama-3.3-70b-versatile",')
open('agents/generator_agent.py', 'w').write(content)
print('done')
