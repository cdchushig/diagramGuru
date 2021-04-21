from app.models import Diagram

from bootstrap_modal_forms.forms import BSModalModelForm


class DiagramModelForm(BSModalModelForm):
    class Meta:
        model = Diagram
        fields = ['name', 'author', 'comment']