from django import forms


class UploadFileForm(forms.Form):
    diagram_file = forms.FileField()