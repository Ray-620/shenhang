# Build and start docker compose setup
production:
	APP_ENV=Production cpu_limit=2 docker-compose -f docker-compose.yaml up --build
production_nobuild:
	APP_ENV=Production cpu_limit=2 docker-compose -f docker-compose.yaml up
