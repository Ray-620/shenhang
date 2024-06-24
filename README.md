# This project is for Shenghang
## Tasks covered 
- [x] RAG API

<details>
    <summary>How to start the service</summary>

Run this command to start the service if this's your first time to run this service
```bash
make production
```

If this's not your first time to run this service, this image has been created before and you don't change your Docker file. You can run this to run your service.

```bash
make production_nobuild
```
</details>


<details>
    <summary>How to test</summary>


### API usage statistics
```bash
curl -s -H 'Content-Type: application/json' http://localhost:3556/ai/stats
```
</details>

