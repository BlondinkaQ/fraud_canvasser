from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
import joblib

logistic_model_path = os.path.join('models', 'logistic_regression_model.pkl')
forest_model_path = os.path.join('models', 'random_forest_model.pkl')

logistic_model = joblib.load(logistic_model_path)
forest_model = joblib.load(forest_model_path)


def index(request):
    if request.user.is_authenticated:
        # User is logged in, show the home page content
        return render(request, 'index.html')
    else:
        if request.method == 'POST':
            form = AuthenticationForm(request, data=request.POST)
            if form.is_valid():
                username = form.cleaned_data.get('username')
                password = form.cleaned_data.get('password')
                user = authenticate(username=username, password=password)
                if user is not None:
                    login(request, user)
                    messages.info(request, f"You are now logged in as {username}.")
                    return redirect('index')
                else:
                    messages.error(request, "Invalid username or password.")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            form = AuthenticationForm()
        return render(request, 'login.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                return redirect('index')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect('index')

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect('index')
        else:
            messages.error(request, "Unsuccessful registration. Invalid information.")
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})


def process_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        model_name = request.POST.get('model')

        if csv_file:
            data = pd.read_csv(csv_file)
            file_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_data.csv')
            data.to_csv(file_path, index=False)

            if model_name == 'logistic':
                model_path = os.path.join(settings.BASE_DIR, 'myapp', 'models', 'logistic_regression_model.pkl')
            elif model_name == 'forest':
                model_path = os.path.join(settings.BASE_DIR, 'myapp', 'models', 'random_forest_model.pkl')
            else:
                return HttpResponse("Model name not recognized.")

            data = data.dropna(subset=[data.columns[0]])

            data['Resolved'] = data['Resolved'].astype(int)

            data['Canvass Date'] = pd.to_datetime(data['Canvass Date'], errors='coerce')
            data['Flag Date'] = pd.to_datetime(data['Flag Date'], errors='coerce')
            data['Canvass Date'] = data['Canvass Date'].apply(lambda x: x.toordinal() if pd.notnull(x) else 0)
            data['Flag Date'] = data['Flag Date'].apply(lambda x: x.toordinal() if pd.notnull(x) else 0)

            X = data[['Office', 'Canvass Date', 'Canvasser', 'Flag Date', 'Flag']]
            y = data['Resolved']

            X = pd.get_dummies(X, drop_first=True)
            X.fillna(0, inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred))
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy * 100:.2f}%')

            loaded_model = joblib.load(model_path)
            #print(f'Model loaded from {joblib_file}')

            y_pred_loaded = loaded_model.predict(X_test)
            print(classification_report(y_test, y_pred_loaded))
            accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
            print(f'Accuracy of loaded model: {accuracy_loaded * 100:.2f}%')

            total_records = len(data)
            fraud_records = sum(data['Resolved'])
            fraud_percentage = (fraud_records / total_records) * 100
            print(f'Fraud Percentage: {fraud_percentage:.2f}%')

            data['Fraud Percentage'] = data['Resolved'] * 100

            output_file_path = os.path.join(settings.BASE_DIR, 'downloads',
                                            'Flag_Responses_with_Fraud_Percentage_1.csv')
            data.to_csv(output_file_path, index=False)

            return render(request, 'result.html', {
                'data': data.to_html(classes='table table-striped', index=False),
                'accuracy': accuracy,
                'fraud_percentage': fraud_percentage
            })

        return HttpResponse("No CSV file uploaded.")

    return HttpResponse("Invalid request method.")


def download_csv(request):
    file_path = os.path.join(settings.BASE_DIR, 'downloads', 'Flag_Responses_with_Fraud_Percentage_1.csv')

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    else:
        return HttpResponse('File not found', status=404)