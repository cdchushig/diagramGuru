diagramGuruTool
=================

diagramGuruTool is an analytical tool that evaluates diagrams according to different Modeling skills.

#### Run Django application
```
python manage.py runserver 8080
```

#### Export env vars
```
export $(cat .env)
```

### Docker commands
The next Docker commands are used to deploy the Django application.

#### Docker build containers
```
docker-compose -f docker-compose.yml down -v
docker-compose -f docker-compose.yml up -d --build
```

#### Docker commands for making migrations
```
docker-compose exec web python manage.py makemigrations
docker-compose exec web python manage.py migrate
```

#### Docker commands for translations and static files
```
docker-compose exec web python manage.py flush --no-input
docker-compose exec web python manage.py collectstatic --no-input --clear
docker-compose exec web python manage.py makemessages -l tr
docker-compose exec web python manage.py compilemessages
```

#### Docker commands for accessing to database

Build image with all dependencies:
```shell script
docker image build -t diagram_guru_web .
```

Run container in bg:
```shell script
docker run -p 8080:8080 -d diagram_guru_web
```

Access to console of container:
```shell script
docker run -it diagram_guru_web
```

Access to database
```shell script
$ docker run -d -p 3306:3306 --name drscratchv3_database -e MYSQL_ROOT_PASSWORD=password mysql
$ docker exec -it <containerid> mysql -p
```


#### Access to terminal
```
$ docker exec -it <containerid> bash
```

### Create image annotations

To improve the model, other images can be used in the training process. To get images for training and
validation, an application for labeling images is needed. [LabelImg](https://github.com/tzutalin/labelImg), a graphical image annotation tool, will be used in this project.

