from django.shortcuts import render


def index_app(request):
    return render(request, 'index.html')
