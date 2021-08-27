diagramGuruTool
=================

diagramGuruTool is an analytical tool that evaluates diagrams according to different Modeling skills.

```
python manage.py runserver 8080
```

### Docker commands
The next Docker commands are used for deploying the application.

```
docker-compose -f docker-compose.yml down -v
docker-compose -f docker-compose.yml up -d --build
docker-compose exec web python manage.py flush --no-input
docker-compose exec web python manage.py collectstatic --no-input --clear
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py makemessages -l tr
docker-compose exec web python manage.py compilemessages
```

### Access to containers
```
$ docker run -d -p 3306:3306 --name drscratchv3_database -e MYSQL_ROOT_PASSWORD=password mysql
$ docker exec -it <containerid> mysql -p
$ docker exec -it <containerid> bash
```

### Create image annotations

To improve the model, other images can be used for the training process. In order to get images for training and
validation. An application for getting labeled images is needed. [LabelImg](https://github.com/tzutalin/labelImg) is a 
graphical image annotation tool.

### Deploying the application with ansible and vagrant

#### ssh to the box
```
$ vagrant ssh
```
#### Provisioning the box
```
$ vagrant provision
```
#### Reboot the box
```
$ vagrant reload
```
#### Shutdown the box
```
$ vagrant halt
```