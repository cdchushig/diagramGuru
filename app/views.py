import os
from datetime import datetime
import base64
import json
import requests

from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

from app.forms import UploadFileForm, NewUserForm, NewModelForm
from app.models import Diagram
from app.app_consts import Consts
from app.diagram_transfomer import create_str_xml_file, create_list_shape_edges_objects, get_diagram_image_bytes

from PIL import Image
from pdf2image import convert_from_path

import logging
import coloredlogs

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

PATH_DIAGRAM_GURU_PROJECT = os.path.dirname(os.path.dirname(__file__))
PATH_DIR_UPLOADS = PATH_DIAGRAM_GURU_PROJECT + "/uploads/"
PATH_DIR_DIAGRAMS = PATH_DIAGRAM_GURU_PROJECT + '/diagrams/'


def index_app(request):
    # username = request.POST['username']
    # password = request.POST['password']
    # user = authenticate(request, username=username, password=password)
    # if user is not None:
    #     login(request, user)
    #     return render(request, 'index2.html')
    # else:
    #     return render(request, 'index2.html')
    return render(request, 'index2.html')


def login_request(request):

    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                return redirect("/list_diagram")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")

    form = AuthenticationForm()

    return render(request=request, template_name="login.html", context={"login_form": form})


def signup(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful." )
            return redirect("/list_diagram")

        messages.error(request, "Unsuccessful registration. Invalid information.")

    form = NewUserForm()

    return render(request=request, template_name="signup.html", context={"register_form": form})


def save_model(request):

    if request.method == "POST":
        form = NewModelForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect("/list_diagram")

        messages.error(request, "Unsuccessful registration. Invalid information.")

    form = NewUserForm()

    return render(request=request, template_name="save_model.html", context={"register_form": form})


def show_dashboard(request):
    return render(request, "dashboard.html", context={})


def list_diagram(request):
    logger.info('List diagrams')
    list_diagram = Diagram.objects.all()
    return render(request, 'list_diagram.html', {"bpmn_list": list_diagram})


def show_model(request):
    """
    Handle Request POST with uploded file
    :param request: Request object.
    :return:
    """
    if request.method == 'POST':
        upload_file_form = UploadFileForm(request.POST, request.FILES)
        if upload_file_form.is_valid():
            diagram_name_unique_id = handle_uploaded_file(request.FILES['diagram_file'])
            request.session['diagram_name'] = diagram_name_unique_id
            return HttpResponseRedirect('/open_external_diagram')
    else:
        return HttpResponseRedirect('/')


def handle_404(request):
    data = {}
    return render(request, 'error.html', data)


def do_request_to_api_diagram_detector(img_bytes, diagram_filename_uploaded_uid_with_ext: str) -> dict:
    """
    Do request to API diagram detector.
    :param img_bytes:
    :param diagram_filename_uploaded_uid_with_ext:
    :return:
    """
    diagram_filename = diagram_filename_uploaded_uid_with_ext.split('/')[-1]
    im_b64 = base64.b64encode(img_bytes).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_b64, "diagram_name": diagram_filename})
    response = requests.post(Consts.API_URL, data=payload, headers=headers)

    dict_objects = {}

    try:
        dict_objects = response.json()
    except requests.exceptions.RequestException:
        logger.error('Error in response from api')
        logger.error(response.text)

    return dict_objects


def convert_upload_format_file_to_png(input_path, output_path, ext_file):

    if ext_file in 'pdf':
        img_file = convert_from_path(input_path + 'example.pdf')
    elif ext_file in 'jpg':
        img_file = Image.open(input_path + 'PhotoName' + ext_file)
        img_file.save(output_path + 'PhotoName.png' + '.png')


def save_file_uploaded(diagram_file_uploaded, diagram_path_filename_uploaded):
    """
    Save diagram file to local folder.
    :param diagram_file_uploaded:
    :param diagram_path_filename_uploaded:
    :return:
    """
    with open(diagram_path_filename_uploaded, 'wb+') as destination:
        for chunk in diagram_file_uploaded.chunks():
            destination.write(chunk)


def get_filename_uid_and_ext(diagram_filename_with_extension):
    """
    Get a filename with an unique id
    :param diagram_filename_with_extension: String object. Name of file uploaded with extension.
    :return: String object.
    """
    unique_id = datetime.now().strftime('_%Y_%m_%d_%H%M%S')
    list_diagram_filename = diagram_filename_with_extension.split('.')
    ext_file = '.' + list_diagram_filename[1]
    diagram_filename_unique_id = list_diagram_filename[0] + unique_id
    return diagram_filename_unique_id, ext_file


def handle_uploaded_file(diagram_file_uploaded):
    diagram_filename_uid, ext_file = get_filename_uid_and_ext(diagram_file_uploaded.name)
    logger.info('Diagram file name: %s', diagram_filename_uid)
    path_diagram_filename_uid = PATH_DIR_UPLOADS + diagram_filename_uid + ext_file
    save_file_uploaded(diagram_file_uploaded, path_diagram_filename_uid)
    # Read and resize and image
    img_bytes, diagram_img_resized = get_diagram_image_bytes(path_diagram_filename_uid, ext_file)
    # Do request to api
    dict_diagram_objects = do_request_to_api_diagram_detector(img_bytes, path_diagram_filename_uid)
    lst_diagram_shapes, lst_diagram_edges = create_list_shape_edges_objects(diagram_img_resized, dict_diagram_objects)

    if len(dict_diagram_objects) != 0:
        xml_str = create_str_xml_file(lst_diagram_shapes, lst_diagram_edges, diagram_filename_uid)
    else:
        logger.error('No objects in diagram recognized!')

    return diagram_filename_uid


def delete_diagram(request, id):
    try:
        qs = Diagram.objects.filter(id=id)
        if qs.exists():
            name = qs[0].name
            msg = u"Diagram '%s' deleted successfully!" % name
            qs.delete()
        else:
            msg = None
            tag_msg = None
        tag_msg = 'success'
    except Exception as err:
        msg = err.message
        tag_msg = 'error'
    bpmn_list = Diagram.objects.all()
    context = {'bpmn_list':bpmn_list, 'msg': msg, 'tag_msg': tag_msg}
    template = 'bpmndesigner/list.html'
    return render(request, template, context)


def open_diagram(request, id):
    try:
        qs = Diagram.objects.filter(id=id)
        if qs.exists():
            bpmn = qs[0]
            logger.info(bpmn)
            bpmn_file_content = bpmn.xml_content
            context = {'bpmn_file_content': bpmn_file_content, 'id_bpmn': bpmn.id}
            return render(request, 'bpmn/modeler.html', context)

    except Exception as err:
        logger.error('Exception')
        logger.error(err)


def create_model(request):
    bpmn_filename = os.path.join(settings.BASE_DIR, 'static', 'app', 'diagrams', 'blank-diagram.xml')
    with open(bpmn_filename, 'r') as f:
        bpmn_file_content = f.read()
    context = {'bpmn_filename': bpmn_filename, 'bpmn_file_content': bpmn_file_content, 'id_bpmn': -1}
    template = 'modeler/modeler_creator.html'
    return render(request, template, context)


def save_diagram(request):
    try:
        id = request.POST.get("id")
        name = request.POST.get("name")
        xml_content = request.POST.get("xml_content")

        if id and id != '-1': # if id was given, then update the diagram and don't create a new one
            qs = Diagram.objects.filter(id=id)
            if qs.exists():
                bpmn = qs[0]
                bpmn.xml_content = xml_content
                bpmn.save()
                result_msg = "BPMN updated sucessfully!"
                result_status = 2  # TODO: create an enum or choices to hold this status values
        else:
            # create a new diagram
            bpmn = Diagram.objects.create(name=name, xml_content=xml_content)
            bpmn.save()
            result_msg = "BPMN saved sucessfully!"
            result_status = 1  # TODO: create an enum or choices to hold this status values

    except Exception as err:
        logger.error(err)
        result_msg = err.message
        result_status = 0

    return HttpResponse(content_type="application/json", content='{"status":"%d", "msg":"%s"}' % (result_status, result_msg))


def open_external_diagram(request):
    bpmn_filename_xml = request.session.get('diagram_name')
    bpmn_filename = PATH_DIR_DIAGRAMS + bpmn_filename_xml + '.xml'

    with open(bpmn_filename, 'r') as f:
        bpmn_file_content = f.read()

    context = {'bpmn_filename': bpmn_filename, 'bpmn_file_content': bpmn_file_content, 'id_bpmn': -1}
    template = 'modeler/modeler_oc.html'
    return render(request, template, context)
