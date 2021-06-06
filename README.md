diagramGuruTool
=================

diagramGuruTool is an analytical tool that evaluates diagrams according to different Modeling skills.

```
python manage.py runserver 8080
```

Build the Docker image
```
docker build --tag django_diagram_guru:latest .
```

Create and run the Docker container
```
docker run --name django_diagram_guru -d -p 8000:8000 django_diagram_guru:latest
``` 

Utils commands
```
docker container ps
```