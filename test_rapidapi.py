import asyncio
import httpx

async def test():
    client = httpx.AsyncClient()
    res = await client.get('https://youtube-mp36.p.rapidapi.com/dl?id=QClGhSzBvwU', headers={'X-RapidAPI-Key':'c7031c7a99msh08290c94701ded7p16426cjsnbb555e837ab8','X-RapidAPI-Host':'youtube-mp36.p.rapidapi.com'})
    print('Status:', res.status_code)
    try:
        data = res.json()
        download_url = data.get('link')
        print('DL URL:', download_url)
        if download_url:
            stream_res = await client.get(download_url, follow_redirects=True)
            print('Stream Status:', stream_res.status_code)
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
