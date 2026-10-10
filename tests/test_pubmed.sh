python3 -c "
from src.kb.pubmed_client import PubMedClient                                                                        
import asyncio

async def test():
    client = PubMedClient(api_key='168ef570fb9e51dc79d0cee08bd779957909')
    results = await client.search('pneumothorax chest xray', max_results=3)
    print(f'Results: {len(results)}')
    for r in results:
        print(f'  - {r}')

asyncio.run(test())
"