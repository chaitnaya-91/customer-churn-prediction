from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
import re
import logging, traceback
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
from datetime import datetime
import io
import csv
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['churn_prediction_db']
users_collection = db['users']
customers_collection = db['customers']
predictions_collection = db['predictions']


def get_dataset_filename(industry):
    try:
        col = db[f"{industry}_dataset_state"]
        state = col.find_one({'user_id': session.get('user_id')})
        if state:
            return state.get('filename')
    except Exception:
        pass
    return None


# Churn Prediction Logic (Simple Rule-Based System)
def predict_churn(tenure, monthly_charges, total_charges, contract_type, payment_method):
    """
    Simple rule-based churn prediction
    Returns: (churn_risk, risk_percentage, reasons)
    """
    risk_score = 0
    reasons = []
    
    # Tenure factor (less tenure = higher risk)
    if tenure < 12:
        risk_score += 30
        reasons.append("Low tenure (< 12 months)")
    elif tenure < 24:
        risk_score += 15
        reasons.append("Medium tenure (12-24 months)")
    
    # Monthly charges factor (high charges = higher risk)
    if monthly_charges > 80:
        risk_score += 25
        reasons.append("High monthly charges (> $80)")
    elif monthly_charges > 50:
        risk_score += 10
        reasons.append("Moderate monthly charges ($50-$80)")
    
    # Contract type factor (month-to-month = higher risk)
    if contract_type == 'Month-to-month':
        risk_score += 20
        reasons.append("Month-to-month contract")
    elif contract_type == 'One year':
        risk_score += 5
        reasons.append("One year contract")
    
    # Payment method factor (electronic check = higher risk)
    if payment_method == 'Electronic check':
        risk_score += 15
        reasons.append("Electronic check payment")
    
    # Determine risk level
    if risk_score > 60:
        churn_risk = "High"
        risk_color = "red"
    elif risk_score > 30:
        churn_risk = "Medium"
        risk_color = "yellow"
    else:
        churn_risk = "Low"
        risk_color = "green"
    
    return churn_risk, risk_score, reasons, risk_color


# Login required decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_role' not in session or session.get('user_role') != 'admin':
            flash('Admin access required', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('home.html')


@app.route('/home')
def home():
    if 'user_id' in session:
        # Get quick stats for logged-in users
        total_customers = customers_collection.count_documents({})
        recent_predictions = list(predictions_collection.find(
            {'user_id': session['user_id']}
        ).sort('created_at', -1).limit(3))
        return render_template('home.html', 
                             total_customers=total_customers,
                             recent_predictions=recent_predictions)
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not first_name or not last_name or not email or not password:
            flash('All fields are required', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')

        # Password strength: min 6 chars, one uppercase, one digit, one symbol
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'danger')
            return render_template('register.html')
        if not re.search(r'[A-Z]', password):
            flash('Password must include at least one uppercase letter', 'danger')
            return render_template('register.html')
        if not re.search(r'\d', password):
            flash('Password must include at least one digit', 'danger')
            return render_template('register.html')
        if not re.search(r'[^A-Za-z0-9]', password):
            flash('Password must include at least one symbol (e.g. @#$%&*)', 'danger')
            return render_template('register.html')
        
        # Check if user already exists
        if users_collection.find_one({'email': email}):
            flash('Email already registered', 'danger')
            return render_template('register.html')
        
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create user document
        user = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': hashed_password,
            'role': 'user',  # Default role
            'created_at': datetime.now()
        }
        
        # Insert user
        result = users_collection.insert_one(user)
        
        # Auto-login after registration
        session['user_id'] = str(result.inserted_id)
        session['user_name'] = f"{first_name} {last_name}"
        session['user_role'] = 'user'
        
        flash('Registration successful! Welcome aboard.', 'success')
        return redirect(url_for('home'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Find user
        user = users_collection.find_one({'email': email})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user_id'] = str(user['_id'])
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            session['user_role'] = user.get('role', 'user')
            
            flash(f'Welcome back, {user["first_name"]}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent predictions for this user
    recent_predictions = list(predictions_collection.find(
        {'user_id': session['user_id']}
    ).sort('created_at', -1).limit(5))
    
    # Get stats
    total_customers = customers_collection.count_documents({})
    total_predictions = len(recent_predictions)
    return render_template('dashboard.html', 
                         recent_predictions=recent_predictions,
                         total_customers=total_customers,
                         total_predictions=total_predictions)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        customer_id = request.form.get('customer_id')
        tenure = float(request.form.get('tenure', 0))
        monthly_charges = float(request.form.get('monthly_charges', 0))
        total_charges = float(request.form.get('total_charges', 0))
        contract_type = request.form.get('contract_type')
        payment_method = request.form.get('payment_method')
        
        # Make prediction
        churn_risk, risk_score, reasons, risk_color = predict_churn(
            tenure, monthly_charges, total_charges, contract_type, payment_method
        )
        
        # Save prediction
        prediction_doc = {
            'user_id': session['user_id'],
            'customer_id': customer_id,
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'dataset_filename': None,
            'churn_risk': churn_risk,
            'risk_score': risk_score,
            'reasons': reasons,
            'created_at': datetime.now()
        }
        predictions_collection.insert_one(prediction_doc)
        
        flash(f'Churn Risk: {churn_risk} ({risk_score}%)', f'{risk_color}')
        return render_template('predict.html', 
                             churn_risk=churn_risk,
                             risk_score=risk_score,
                             reasons=reasons,
                             risk_color=risk_color)
    
    return render_template('predict.html')


@app.route('/models')
@login_required
def models():
    return render_template('models.html')


@app.route('/home_button')
@login_required
def home_button():
    # Redirect admin to admin panel, others to dashboard
    if session.get('user_role') == 'admin':
        return redirect(url_for('admin_panel'))
    return redirect(url_for('dashboard'))


@app.route('/admin')
@login_required
@admin_required
def admin_panel():
    users = list(users_collection.find({}).sort('created_at', -1))
    industries = ['telecom','banking','ecommerce','subscription','gym','rideshare']
    datasets = {}
    for ind in industries:
        try:
            col = db.get_collection(f"{ind}_dataset_state")
            items = list(col.find({}).sort('updated_at', -1))
            datasets[ind] = items
        except Exception:
            datasets[ind] = []

    total_preds = predictions_collection.count_documents({})
    recent_preds = list(predictions_collection.find({}).sort('created_at', -1).limit(200))
    # Load dataset submission requests for admin review
    try:
        submissions = list(db.get_collection('dataset_submissions').find({}).sort('created_at', -1).limit(200))
    except Exception:
        submissions = []

    try:
        memberships = list(db.get_collection('memberships').find({}).sort('created_at', -1))
    except Exception:
        memberships = []

    return render_template('admin_panel.html', users=users, datasets=datasets, total_preds=total_preds, recent_preds=recent_preds, submissions=submissions, memberships=memberships)


@app.route('/admin/user/delete/<user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    try:
        users_collection.delete_one({'_id': ObjectId(user_id)})
        flash('User deleted', 'success')
    except Exception:
        flash('Error deleting user', 'danger')
    return redirect(url_for('admin_panel'))


@app.route('/admin/user/toggle_role/<user_id>', methods=['POST'])
@login_required
@admin_required
def admin_toggle_role(user_id):
    try:
        u = users_collection.find_one({'_id': ObjectId(user_id)})
        if u:
            current = u.get('role', 'user')
            # Cycle: user -> gold -> premium -> user (never toggle to admin via this)
            if current == 'user':
                new_role = 'gold'
            elif current == 'gold':
                new_role = 'premium'
            else:
                new_role = 'user'
            users_collection.update_one({'_id': ObjectId(user_id)}, {'$set': {'role': new_role}})
            flash(f'User plan updated to {new_role.title()}', 'success')
    except Exception:
        flash('Error updating role', 'danger')
    return redirect(url_for('admin_panel'))


@app.route('/admin/membership/download/<membership_id>')
@login_required
@admin_required
def admin_download_membership_screenshot(membership_id):
    try:
        col = db.get_collection('memberships')
        doc = col.find_one({'_id': ObjectId(membership_id)})
        if not doc:
            flash('Membership not found', 'warning')
            return redirect(url_for('admin_panel'))
        screenshot = doc.get('payment_screenshot')
        if not screenshot:
            flash('No payment screenshot on file', 'warning')
            return redirect(url_for('admin_panel'))
        file_path = os.path.join('static', 'payments', screenshot)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=screenshot)
        else:
            flash('File not found on server', 'danger')
            return redirect(url_for('admin_panel'))
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('admin_panel'))


@app.route('/admin/membership/approve/<membership_id>', methods=['POST'])
@login_required
@admin_required
def admin_approve_membership(membership_id):
    try:
        col = db.get_collection('memberships')
        doc = col.find_one({'_id': ObjectId(membership_id)})
        if not doc:
            flash('Membership not found', 'warning')
            return redirect(url_for('admin_panel'))
        current_status = doc.get('approved', False)
        col.update_one({'_id': ObjectId(membership_id)}, {'$set': {'approved': not current_status}})
        # Also update user's role to match the plan if approving
        if not current_status:
            plan = doc.get('plan', 'gold')
            user_id = doc.get('user_id')
            if user_id:
                users_collection.update_one({'_id': ObjectId(user_id)}, {'$set': {'role': plan}})
        flash('Membership approval status updated', 'success')
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('admin_panel'))


@app.route('/admin/membership/reject/<membership_id>', methods=['POST'])
@login_required
@admin_required
def admin_reject_membership(membership_id):
    try:
        col = db.get_collection('memberships')
        doc = col.find_one({'_id': ObjectId(membership_id)})
        if not doc:
            flash('Membership not found', 'warning')
            return redirect(url_for('admin_panel'))
        # Remove the screenshot file if it exists
        screenshot = doc.get('payment_screenshot')
        if screenshot:
            file_path = os.path.join('static', 'payments', screenshot)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        # Delete the membership request
        col.delete_one({'_id': ObjectId(membership_id)})
        flash('Membership request rejected and removed', 'success')
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('admin_panel'))


@app.route('/admin/dataset/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_dataset():
    industry = request.form.get('industry')
    state_id = request.form.get('state_id')
    if not industry or not state_id:
        flash('Missing parameters', 'danger')
        return redirect(url_for('admin_panel'))
    try:
        col = db.get_collection(f"{industry}_dataset_state")
        doc = col.find_one({'_id': ObjectId(state_id)})
        if doc:
            # remove state doc
            col.delete_one({'_id': ObjectId(state_id)})
            # attempt to remove local saved dataset file if present
            fn = doc.get('filename')
            if fn:
                import os
                local = os.path.join('dump models', f"{industry}_last_dataset.csv")
                if os.path.exists(local):
                    try:
                        os.remove(local)
                    except Exception:
                        pass
            flash('Dataset entry removed', 'success')
        else:
            flash('Dataset not found', 'warning')
    except Exception as e:
        flash(f'Error removing dataset: {str(e)}', 'danger')
    return redirect(url_for('admin_panel'))


@app.route('/admin/dataset/download')
@login_required
@admin_required
def admin_download_dataset():
    industry = request.args.get('industry')
    state_id = request.args.get('state_id')
    if not industry or not state_id:
        flash('Missing parameters for download', 'danger')
        return redirect(url_for('admin_panel'))
    try:
        col = db.get_collection(f"{industry}_dataset_state")
        doc = col.find_one({'_id': ObjectId(state_id)})
        if not doc:
            flash('Dataset state not found', 'warning')
            return redirect(url_for('admin_panel'))

        # Prefer explicit stored file path, otherwise fallback to conventional path
        file_path = doc.get('file_path') or os.path.join('dump models', f"{industry}_last_dataset.csv")
        filename = doc.get('filename') or f"{industry}_dataset.csv"
        if file_path and os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            flash('File not available on server', 'danger')
            return redirect(url_for('admin_panel'))
    except Exception as e:
        flash(f'Error downloading dataset: {str(e)}', 'danger')
        return redirect(url_for('admin_panel'))


@app.route('/admin/submission/download/<submission_id>')
@login_required
@admin_required
def admin_download_submission(submission_id):
    try:
        col = db.get_collection('dataset_submissions')
        doc = col.find_one({'_id': ObjectId(submission_id)})
        if not doc:
            flash('Submission not found', 'warning')
            return redirect(url_for('admin_panel'))

        file_path = doc.get('file_path')
        filename = doc.get('filename') or 'dataset.csv'
        if file_path and os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            flash('File not available on server', 'danger')
            return redirect(url_for('admin_panel'))
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('admin_panel'))


@app.route('/admin/submission/delete/<submission_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_submission(submission_id):
    try:
        col = db.get_collection('dataset_submissions')
        doc = col.find_one({'_id': ObjectId(submission_id)})
        if not doc:
            flash('Submission not found', 'warning')
            return redirect(url_for('admin_panel'))

        # Remove file if exists
        file_path = doc.get('file_path')
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        col.delete_one({'_id': ObjectId(submission_id)})
        flash('Submission removed', 'success')
    except Exception as e:
        flash(f'Error deleting submission: {str(e)}', 'danger')
    return redirect(url_for('admin_panel'))


@app.route('/submit-dataset', methods=['GET', 'POST'])
@login_required
def submit_dataset():
    """User-facing dataset submission form for business requests."""
    # Restrict to admin and premium users
    if session.get('user_role') not in ['admin', 'premium']:
        flash('Only Admin and Premium members can submit datasets.', 'danger')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        city = request.form.get('city')
        outcome = request.form.get('outcome')
        questions = request.form.get('questions')
        expected = request.form.get('expected')
        notes = request.form.get('notes')

        # Basic validation for required fields
        if not full_name or not email or not phone or not city or not outcome:
            flash('Please fill all required fields marked with *', 'danger')
            return render_template('submit_dataset.html', form=request.form)

        # Handle uploaded file
        csv_file = request.files.get('dataset_file')
        saved_filename = None
        saved_path = None
        try:
            if csv_file and csv_file.filename != '' and csv_file.filename.lower().endswith('.csv'):
                os.makedirs(os.path.join('dump models', 'submissions'), exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                safe_name = f"{timestamp}_{csv_file.filename}"
                saved_path = os.path.join('dump models', 'submissions', safe_name)
                csv_file.save(saved_path)
                saved_filename = csv_file.filename

            # Save submission to MongoDB
            sub_doc = {
                'full_name': full_name,
                'email': email,
                'phone': phone,
                'city': city,
                'outcome': outcome,
                'questions': questions,
                'expected': expected,
                'notes': notes,
                'filename': saved_filename,
                'file_path': saved_path,
                'user_id': session.get('user_id'),
                'created_at': datetime.now()
            }
            db.get_collection('dataset_submissions').insert_one(sub_doc)
            flash('Dataset request submitted — admin will review it shortly.', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Error submitting dataset request: {str(e)}', 'danger')
            return render_template('submit_dataset.html', form=request.form)

    return render_template('submit_dataset.html')


# Industry-specific prediction pages (placeholder routes)
@app.route('/predict/telecom')
@login_required
def predict_telecom():
    return redirect(url_for('telecom_upload'))


@app.route('/predict/telecom/customer', methods=['GET', 'POST'])
@login_required
def telecom_customer_predict():
    result = None
    form   = None

    if request.method == 'POST':
        # Collect all form fields
        form = request.form.to_dict()

        def safe_float(key, default=0.0):
            try:
                return float(form.get(key, default) or default)
            except (ValueError, TypeError):
                return default

        tenure          = safe_float('tenure')
        monthly_charges = safe_float('monthlycharges')
        total_charges   = safe_float('totalcharges')
        datausage       = safe_float('datausage')
        customersupportcalls = safe_float('customersupportcalls')
        complaints      = safe_float('complaints')
        lastrechargedaysago = safe_float('lastrechargedaysago')
        contract_type   = form.get('contract', 'Month-to-month')
        payment_method  = form.get('paymentmethod', '')
        internetservice = form.get('internetservice', '')
        autopay         = 1 if form.get('autopay') == '1' else 0

        # Attempt to load trained model
        import os, joblib
        import pandas as pd
        models_dir = 'dump models'

        # Support common filenames used when saving models
        model_candidates = [
            'telecom_best_model.pkl',
            'telecom_svm_model.pkl',
            'telecom_svm.pkl',
            'telecom_model.pkl'
        ]
        scaler_candidates = [
            'telecom_scaler.pkl',
            'telecom_standardscaler.pkl'
        ]

        model_path = None
        for fn in model_candidates:
            p = os.path.join(models_dir, fn)
            if os.path.exists(p):
                model_path = p
                break

        scaler_path = None
        for fn in scaler_candidates:
            p = os.path.join(models_dir, fn)
            if os.path.exists(p):
                scaler_path = p
                break

        features_path = os.path.join(models_dir, 'telecom_features.pkl')

        if model_path and scaler_path:
            best_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Load feature columns if available (recommended). Otherwise try to infer.
            if os.path.exists(features_path):
                feature_cols = joblib.load(features_path)
            else:
                # Try to use scaler metadata if present
                feature_cols = None
                if hasattr(scaler, 'feature_names_in_'):
                    try:
                        feature_cols = list(getattr(scaler, 'feature_names_in_'))
                    except Exception:
                        feature_cols = None

                if feature_cols is None:
                    # Fallback common telecom features order (numeric + dummies)
                    fallback = [
                        'tenure','monthlycharges','totalcharges','datausage',
                        'customersupportcalls','complaints','lastrechargedaysago','autopay',
                        'internetservice_DSL','internetservice_Fiber'
                    ]
                    # If scaler knows number of inputs, trim or extend
                    if hasattr(scaler, 'n_features_in_'):
                        n = getattr(scaler, 'n_features_in_')
                        feature_cols = fallback[:n]
                    else:
                        feature_cols = fallback

            # Create dataframe for single prediction
            input_data = pd.DataFrame(0, index=[0], columns=feature_cols)

            # Fill numeric values
            if 'tenure' in input_data.columns: input_data['tenure'] = tenure
            if 'monthlycharges' in input_data.columns: input_data['monthlycharges'] = monthly_charges
            if 'totalcharges' in input_data.columns: input_data['totalcharges'] = total_charges
            if 'datausage' in input_data.columns: input_data['datausage'] = datausage
            if 'customersupportcalls' in input_data.columns: input_data['customersupportcalls'] = customersupportcalls
            if 'complaints' in input_data.columns: input_data['complaints'] = complaints
            if 'lastrechargedaysago' in input_data.columns: input_data['lastrechargedaysago'] = lastrechargedaysago
            if 'autopay' in input_data.columns: input_data['autopay'] = autopay

            # Handle categorical features encoding manually based on what get_dummies might have produced
            is_col = f'internetservice_{internetservice}'
            if is_col in input_data.columns:
                input_data[is_col] = 1

            # Attempt to scale the input; fall back gracefully if scaling fails
            try:
                input_scaled = scaler.transform(input_data)
            except Exception:
                try:
                    # Try transforming only numeric subset if scaler expects numeric-only
                    if hasattr(scaler, 'feature_names_in_'):
                        cols = [c for c in getattr(scaler, 'feature_names_in_') if c in input_data.columns]
                        input_scaled = scaler.transform(input_data[cols])
                    else:
                        numeric_cols = [c for c in input_data.columns if input_data[c].dtype.kind in 'biufc']
                        input_scaled = scaler.transform(input_data[numeric_cols])
                except Exception:
                    # As a last resort, don't scale and use raw inputs
                    input_scaled = input_data.values

            # Predict using probability if available, otherwise class prediction
            try:
                if hasattr(best_model, 'predict_proba'):
                    prob = best_model.predict_proba(input_scaled)[0]
                    churn_prob = prob[1] * 100
                else:
                    pred = best_model.predict(input_scaled)[0]
                    churn_prob = 100.0 if int(pred) == 1 else 0.0
            except Exception:
                # If model refuses due to shape mismatch, fallback to rule-based
                churn_prob = None

            if churn_prob is not None:
                risk_score = float(round(churn_prob, 2))
                if risk_score > 60:
                    churn_risk, risk_color = "High", "red"
                elif risk_score > 30:
                    churn_risk, risk_color = "Medium", "yellow"
                else:
                    churn_risk, risk_color = "Low", "green"

                reasons = ["Model predicted based on historical data"]
                best_model_name = str(best_model.__class__.__name__).replace("Classifier", "")

                # Retrieve score from state to show accuracy
                state_col = db['telecom_dataset_state']
                state = state_col.find_one({'user_id': session['user_id']})
                best_accuracy = state.get('results', {}).get('best_score', 0.68) * 100 if state else 68.0
            else:
                # Fallback to rule-based engine if model prediction failed
                churn_risk, risk_score, reasons, risk_color = predict_churn(
                    tenure, monthly_charges, total_charges,
                    contract_type, payment_method
                )
                best_model_name = None
                best_accuracy = None

        else:
            # Fallback to rule-based engine
            churn_risk, risk_score, reasons, risk_color = predict_churn(
                tenure, monthly_charges, total_charges,
                contract_type, payment_method
            )
            best_model_name = None
            best_accuracy = None

        result = {
            'churn_risk':  churn_risk,
            'risk_score':  risk_score,
            'reasons':     reasons,
            'risk_color':  risk_color,
            'best_model_name': best_model_name,
            'best_accuracy': best_accuracy
        }

        # Save to predictions collection
        prediction_doc = {
            'user_id':        session['user_id'],
            'customer_id':    form.get('customer_id', 'Manual Entry'),
            'industry':       'telecom',
            'tenure':         tenure,
            'monthly_charges': monthly_charges,
            'total_charges':  total_charges,
            'contract_type':  contract_type,
            'payment_method': payment_method,
            'dataset_filename': get_dataset_filename('telecom'),
            'churn_risk':     churn_risk,
            'risk_score':     risk_score,
            'reasons':        reasons,
            'created_at':     datetime.now()
        }
        predictions_collection.insert_one(prediction_doc)

    return render_template('telecom_customer_predict.html', result=result, form=form)



@app.route('/predict/telecom/upload', methods=['GET', 'POST'])
@login_required
def telecom_upload():
    import os
    results = None
    filename = None
    state_col = db['telecom_dataset_state']
    
    # Load permanent state
    state = state_col.find_one({'user_id': session['user_id']})
    if state:
        results = state.get('results')
        filename = state.get('filename')

    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        action = request.form.get('action', 'train')

        # Validation
        if not csv_file or csv_file.filename == '':
            flash('No file selected. Please upload a CSV file.', 'danger')
            return render_template('telecom_upload.html', results=results, filename=filename)

        if not csv_file.filename.lower().endswith('.csv'):
            flash('Invalid file type. Only .csv files are accepted.', 'danger')
            return render_template('telecom_upload.html', results=results, filename=filename)

        try:
            # Read CSV content into stream to pass to the functions
            csv_bytes = csv_file.stream.read()
            stream = io.StringIO(csv_bytes.decode('utf-8-sig'), newline=None)

            from telecom_train import train_telecom_models
            
            # Determine if we should only tune XGBoost
            tune_xgboost = (action == 'tune')
            
            # Train models and get results
            results = train_telecom_models(stream, tune_xgboost=tune_xgboost)
            
            # Save the file locally so we can run visualizations on it later
            os.makedirs('dump models', exist_ok=True)
            with open('dump models/telecom_last_dataset.csv', 'wb') as f:
                f.write(csv_bytes)
                
            # Save state to MongoDB
            state_col.update_one(
                {'user_id': session['user_id']},
                {'$set': {
                    'filename': csv_file.filename,
                    'results': results,
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
            filename = csv_file.filename
            
            flash(f'Training complete! Best model: {results["best_model"]} with accuracy {results["best_score"]*100:.2f}%.', 'success')
            return render_template('telecom_upload.html', results=results, filename=filename)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Error during training: {str(e)}', 'danger')
            return render_template('telecom_upload.html', results=results, filename=filename)

    return render_template('telecom_upload.html', results=results, filename=filename)


@app.route('/predict/telecom/visualize', methods=['GET', 'POST'])
@login_required
def telecom_visualize():
    import os
    csv_file = None
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')

    # If GET or no file uploaded in POST, check if we have an active dataset saved locally
    if not csv_file:
        local_path = 'dump models/telecom_last_dataset.csv'
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8-sig') as f:
                    stream = io.StringIO(f.read(), newline=None)
                from telecom_visualize import generate_visualizations
                visualizations, churn_rate, retention_rate = generate_visualizations(stream)
                return render_template('telecom_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
            except Exception as e:
                flash(f'Error reading saved dataset: {str(e)}', 'danger')
                return redirect(url_for('telecom_upload'))
        else:
            flash('No active dataset found to visualize. Please upload one first.', 'danger')
            return redirect(url_for('telecom_upload'))

    # If POST with a new file:
    if not csv_file.filename.lower().endswith('.csv'):
        flash('Invalid file type. Only .csv files are accepted.', 'danger')
        return redirect(url_for('telecom_upload'))

    try:
        stream = io.StringIO(csv_file.stream.read().decode('utf-8-sig'), newline=None)
        from telecom_visualize import generate_visualizations
        visualizations, churn_rate, retention_rate = generate_visualizations(stream)
        return render_template('telecom_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
    except Exception as e:
        flash(f'Error generating visualizations: {str(e)}', 'danger')
        return redirect(url_for('telecom_upload'))


@app.route('/predict/telecom/remove_dataset', methods=['POST'])
@login_required
def telecom_remove_dataset():
    import os
    state_col = db['telecom_dataset_state']
    state_col.delete_one({'user_id': session['user_id']})
    
    # Delete locally saved dataset
    local_path = 'dump models/telecom_last_dataset.csv'
    if os.path.exists(local_path):
        os.remove(local_path)
        
    # Also delete the saved models, scaler, and features so predictions fallback to rule-based!
    model_path = os.path.join('dump models', 'telecom_best_model.pkl')
    scaler_path = os.path.join('dump models', 'telecom_scaler.pkl')
    features_path = os.path.join('dump models', 'telecom_features.pkl')
    
    for path in [model_path, scaler_path, features_path]:
        if os.path.exists(path):
            os.remove(path)
            
    flash('Active dataset and all trained models/operations have been successfully removed.', 'success')
    return redirect(url_for('telecom_upload'))


@app.route('/retrain/<industry>')
@login_required
def retrain(industry):
    # Map industry keys to their upload routes
    mapping = {
        'telecom': 'telecom_upload',
        'banking': 'banking_upload',
        'bank': 'banking_upload',
        'subscription': 'subscription_upload',
        'gym': 'fitness_upload',
        'fitness': 'fitness_upload',
        'rideshare': 'rideshare_upload',
        'ride': 'rideshare_upload',
        'ecommerce': 'ecommerce_upload'
    }
    route_name = mapping.get(industry.lower())
    if route_name:
        flash(f'Retraining started for {industry}. You will be notified when complete.', 'info')
        return redirect(url_for(route_name))
    flash('Unknown industry for retraining', 'danger')
    return redirect(url_for('models'))


@app.route('/subscribe', methods=['POST'])
@login_required
def subscribe():
    name = request.form.get('name') or session.get('user_name')
    email = request.form.get('email') or ''
    city = request.form.get('city') or ''
    plan = request.form.get('plan') or 'gold'
    
    # Handle payment screenshot
    payment_screenshot = request.files.get('payment_screenshot')
    screenshot_filename = None
    if payment_screenshot and payment_screenshot.filename != '':
        os.makedirs(os.path.join('static', 'payments'), exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        screenshot_filename = f"{timestamp}_{payment_screenshot.filename}"
        saved_path = os.path.join('static', 'payments', screenshot_filename)
        payment_screenshot.save(saved_path)

    try:
        col = db.get_collection('memberships')
        doc = {
            'user_id': session.get('user_id'),
            'name': name,
            'email': email,
            'city': city,
            'plan': plan,
            'payment_screenshot': screenshot_filename,
            'created_at': datetime.now()
        }
        col.insert_one(doc)
        return jsonify({'success': True, 'message': 'Membership request submitted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/predict/banking')
@login_required
def predict_banking():
    return redirect(url_for('banking_upload'))


@app.route('/predict/banking/customer', methods=['GET', 'POST'])
@login_required
def banking_customer_predict():
    result = None
    form   = None

    if request.method == 'POST':
        # Collect all form fields
        form = request.form.to_dict()

        def safe_float(key, default=0.0):
            try:
                return float(form.get(key, default) or default)
            except (ValueError, TypeError):
                return default

        creditscore     = safe_float('creditscore', 600.0)
        age             = safe_float('age', 40.0)
        tenure          = safe_float('tenure', 3.0)
        balance         = safe_float('balance', 0.0)
        numofproducts   = safe_float('numofproducts', 1.0)
        hascrcard       = 1 if form.get('hascrcard') == '1' else 0
        isactivemember  = 1 if form.get('isactivemember') == '1' else 0
        estimatedsalary = safe_float('estimatedsalary', 50000.0)
        geography       = form.get('geography', 'France')
        gender          = form.get('gender', 'Female')

        # Attempt to load trained model
        import os, joblib
        import pandas as pd
        models_dir = 'dump models'

        model_candidates = ['banking_best_model.pkl', 'banking_svm_model.pkl', 'banking_model.pkl']
        scaler_candidates = ['banking_scaler.pkl', 'banking_standardscaler.pkl']
        scaler_cols_path = os.path.join(models_dir, 'banking_scaler_cols.pkl')
        features_path = os.path.join(models_dir, 'banking_features.pkl')

        model_path = next((os.path.join(models_dir, f) for f in model_candidates if os.path.exists(os.path.join(models_dir, f))), None)
        scaler_path = next((os.path.join(models_dir, f) for f in scaler_candidates if os.path.exists(os.path.join(models_dir, f))), None)

        if model_path and scaler_path and os.path.exists(features_path):
            best_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_cols = joblib.load(features_path)
            active_num_cols = joblib.load(scaler_cols_path) if os.path.exists(scaler_cols_path) else []

            # Create dataframe for single prediction
            input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
            
            # Fill numeric values
            if 'creditscore' in input_data.columns: input_data['creditscore'] = creditscore
            if 'age' in input_data.columns: input_data['age'] = age
            if 'tenure' in input_data.columns: input_data['tenure'] = tenure
            if 'balance' in input_data.columns: input_data['balance'] = balance
            if 'numofproducts' in input_data.columns: input_data['numofproducts'] = numofproducts
            if 'hascrcard' in input_data.columns: input_data['hascrcard'] = hascrcard
            if 'isactivemember' in input_data.columns: input_data['isactivemember'] = isactivemember
            if 'estimatedsalary' in input_data.columns: input_data['estimatedsalary'] = estimatedsalary

            # Handle categorical features encoding manually
            gender_col = f'gender_{gender}'
            if gender_col in input_data.columns:
                input_data[gender_col] = 1

            geo_col = f'geography_{geography}'
            if geo_col in input_data.columns:
                input_data[geo_col] = 1

            # Scale only the numeric columns that were scaled during training
            if active_num_cols:
                input_data[active_num_cols] = scaler.transform(input_data[active_num_cols])
                
            prob = best_model.predict_proba(input_data)[0]
            churn_prob = prob[1] * 100
            
            risk_score = float(round(churn_prob, 2))
            if risk_score > 60:
                churn_risk, risk_color = "High", "red"
            elif risk_score > 30:
                churn_risk, risk_color = "Medium", "yellow"
            else:
                churn_risk, risk_color = "Low", "green"
                
            reasons = []
            if age > 50:
                reasons.append("Higher age demographic (> 50)")
            if numofproducts > 2:
                reasons.append("High number of banking products (> 2)")
            if isactivemember == 0:
                reasons.append("Inactive bank member status")
            if not reasons:
                reasons = ["Model predicted based on historical behavior patterns"]
            
            best_model_name = str(best_model.__class__.__name__).replace("Classifier", "")
            
            # Retrieve score from state to show accuracy
            state_col = db['banking_dataset_state']
            state = state_col.find_one({'user_id': session['user_id']})
            best_accuracy = state.get('results', {}).get('best_score', 0.68) * 100 if state else 68.0
            
        else:
            # Fallback to rule-based engine
            risk_score = 0
            reasons = []
            
            if age > 50:
                risk_score += 30
                reasons.append("Higher age demographic (> 50)")
            elif age < 30:
                risk_score += 10
                reasons.append("Younger age demographic (< 30)")
                
            if balance < 1000:
                risk_score += 25
                reasons.append("Low account balance (< $1,000)")
                
            if isactivemember == 0:
                risk_score += 20
                reasons.append("Inactive bank member status")
                
            if numofproducts > 2:
                risk_score += 15
                reasons.append("High number of banking products (> 2)")
                
            if risk_score > 60:
                churn_risk, risk_color = "High", "red"
            elif risk_score > 30:
                churn_risk, risk_color = "Medium", "yellow"
            else:
                churn_risk, risk_color = "Low", "green"

            best_model_name = None
            best_accuracy = None

        result = {
            'churn_risk':  churn_risk,
            'risk_score':  risk_score,
            'reasons':     reasons,
            'risk_color':  risk_color,
            'best_model_name': best_model_name,
            'best_accuracy': best_accuracy
        }

        # Save to predictions collection
        prediction_doc = {
            'user_id':        session['user_id'],
            'customer_id':    form.get('customer_id', 'Manual Entry'),
            'surname':        form.get('surname', ''),
            'industry':       'banking',
            'creditscore':    creditscore,
            'geography':      geography,
            'gender':         gender,
            'age':            age,
            'tenure':         tenure,
            'balance':        balance,
            'numofproducts':  numofproducts,
            'hascrcard':      hascrcard,
            'isactivemember': isactivemember,
            'estimatedsalary': estimatedsalary,
            'dataset_filename': get_dataset_filename('banking'),
            'churn_risk':     churn_risk,
            'risk_score':     risk_score,
            'reasons':        reasons,
            'created_at':     datetime.now()
        }
        predictions_collection.insert_one(prediction_doc)

    return render_template('banking_customer_predict.html', result=result, form=form)


@app.route('/predict/banking/upload', methods=['GET', 'POST'])
@login_required
def banking_upload():
    import os
    results = None
    filename = None
    state_col = db['banking_dataset_state']
    
    # Load permanent state
    state = state_col.find_one({'user_id': session['user_id']})
    if state:
        results = state.get('results')
        filename = state.get('filename')

    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        action = request.form.get('action', 'train')

        # Validation
        if not csv_file or csv_file.filename == '':
            flash('No file selected. Please upload a CSV file.', 'danger')
            return render_template('banking_upload.html', results=results, filename=filename)

        if not csv_file.filename.lower().endswith('.csv'):
            flash('Invalid file type. Only .csv files are accepted.', 'danger')
            return render_template('banking_upload.html', results=results, filename=filename)

        try:
            # Read CSV content into stream to pass to the functions
            csv_bytes = csv_file.stream.read()
            stream = io.StringIO(csv_bytes.decode('utf-8-sig'), newline=None)

            from banking_train import train_banking_models
            
            # Determine if we should only tune XGBoost
            tune_xgboost = (action == 'tune')
            
            # Train models and get results
            results = train_banking_models(stream, tune_xgboost=tune_xgboost)
            
            # Save the file locally so we can run visualizations on it later
            os.makedirs('dump models', exist_ok=True)
            with open('dump models/banking_last_dataset.csv', 'wb') as f:
                f.write(csv_bytes)
                
            # Save state to MongoDB
            state_col.update_one(
                {'user_id': session['user_id']},
                {'$set': {
                    'filename': csv_file.filename,
                    'results': results,
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
            filename = csv_file.filename
            
            flash(f'Training complete! Best model: {results["best_model"]} with accuracy {results["best_score"]*100:.2f}%.', 'success')
            return render_template('banking_upload.html', results=results, filename=filename)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Error during training: {str(e)}', 'danger')
            return render_template('banking_upload.html', results=results, filename=filename)

    return render_template('banking_upload.html', results=results, filename=filename)


@app.route('/predict/banking/visualize', methods=['GET', 'POST'])
@login_required
def banking_visualize():
    import os
    csv_file = None
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')

    # If GET or no file uploaded in POST, check if we have an active dataset saved locally
    if not csv_file:
        local_path = 'dump models/banking_last_dataset.csv'
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8-sig') as f:
                    stream = io.StringIO(f.read(), newline=None)
                from banking_visualize import generate_visualizations
                visualizations, churn_rate, retention_rate = generate_visualizations(stream)
                return render_template('banking_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
            except Exception as e:
                flash(f'Error reading saved dataset: {str(e)}', 'danger')
                return redirect(url_for('banking_upload'))
        else:
            flash('No active dataset found to visualize. Please upload one first.', 'danger')
            return redirect(url_for('banking_upload'))

    # If POST with a new file:
    if not csv_file.filename.lower().endswith('.csv'):
        flash('Invalid file type. Only .csv files are accepted.', 'danger')
        return redirect(url_for('banking_upload'))

    try:
        stream = io.StringIO(csv_file.stream.read().decode('utf-8-sig'), newline=None)
        from banking_visualize import generate_visualizations
        visualizations, churn_rate, retention_rate = generate_visualizations(stream)
        return render_template('banking_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
    except Exception as e:
        flash(f'Error generating visualizations: {str(e)}', 'danger')
        return redirect(url_for('banking_upload'))


@app.route('/predict/banking/remove_dataset', methods=['POST'])
@login_required
def banking_remove_dataset():
    import os
    state_col = db['banking_dataset_state']
    state_col.delete_one({'user_id': session['user_id']})
    
    # Delete locally saved dataset
    local_path = 'dump models/banking_last_dataset.csv'
    if os.path.exists(local_path):
        os.remove(local_path)
        
    # Also delete the saved models, scaler, and features so predictions fallback to rule-based!
    model_path = os.path.join('dump models', 'banking_best_model.pkl')
    scaler_path = os.path.join('dump models', 'banking_scaler.pkl')
    scaler_cols_path = os.path.join('dump models', 'banking_scaler_cols.pkl')
    features_path = os.path.join('dump models', 'banking_features.pkl')
    
    for path in [model_path, scaler_path, scaler_cols_path, features_path]:
        if os.path.exists(path):
            os.remove(path)
            
    flash('Active dataset and all trained models/operations have been successfully removed.', 'success')
    return redirect(url_for('banking_upload'))





@app.route('/predict/ecommerce')
@login_required
def predict_ecommerce():
    return redirect(url_for('ecommerce_upload'))

@app.route('/predict/ecommerce/customer', methods=['GET', 'POST'])
@login_required
def ecommerce_customer_predict():
    result = None
    form   = None

    if request.method == 'POST':
        form = request.form.to_dict()
        def safe_float(key, default=0.0):
            try: return float(form.get(key, default) or default)
            except (ValueError, TypeError): return default

        age = safe_float('age', 35.0)
        gender = form.get('gender', 'Female')
        tenure_months = safe_float('tenure_months', 24.0)
        city = form.get('city', 'Mumbai')
        total_orders = safe_float('total_orders', 45.0)
        average_order_value = safe_float('average_order_value', 2500.0)
        days_since_last_order = safe_float('days_since_last_order', 15.0)
        return_rate = safe_float('return_rate', 0.15)
        customer_satisfaction_score = safe_float('customer_satisfaction_score', 7.0)

        import os, joblib
        import pandas as pd
        models_dir = 'dump models'

        model_candidates = [
            'ecommerce_best_model.pkl', 'ecommerce_svm_model.pkl', 'ecommerce_model.pkl'
        ]
        scaler_candidates = ['ecommerce_scaler.pkl', 'ecommerce_standardscaler.pkl']
        features_path = os.path.join(models_dir, 'ecommerce_features.pkl')

        model_path = next((os.path.join(models_dir, f) for f in model_candidates if os.path.exists(os.path.join(models_dir, f))), None)
        scaler_path = next((os.path.join(models_dir, f) for f in scaler_candidates if os.path.exists(os.path.join(models_dir, f))), None)

        if model_path and scaler_path:
            best_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            if os.path.exists(features_path):
                feature_cols = joblib.load(features_path)
            else:
                feature_cols = ['age','tenure_months','total_orders','average_order_value','days_since_last_order','return_rate','customer_satisfaction_score','gender_Female','gender_Male']

            input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
            if 'age' in input_data.columns: input_data['age'] = age
            if 'tenure_months' in input_data.columns: input_data['tenure_months'] = tenure_months
            if 'total_orders' in input_data.columns: input_data['total_orders'] = total_orders
            if 'average_order_value' in input_data.columns: input_data['average_order_value'] = average_order_value
            if 'days_since_last_order' in input_data.columns: input_data['days_since_last_order'] = days_since_last_order
            if 'return_rate' in input_data.columns: input_data['return_rate'] = return_rate
            if 'customer_satisfaction_score' in input_data.columns: input_data['customer_satisfaction_score'] = customer_satisfaction_score

            g_col = f'gender_{gender}'
            if g_col in input_data.columns: input_data[g_col] = 1
            
            c_col = f'city_{city}'
            if c_col in input_data.columns: input_data[c_col] = 1

            try:
                input_scaled = scaler.transform(input_data)
                prob = best_model.predict_proba(input_scaled)[0]
                churn_prob = prob[1] * 100

                risk_score = float(round(churn_prob, 2))
                if risk_score > 60:
                    churn_risk, risk_color = "High", "red"
                elif risk_score > 30:
                    churn_risk, risk_color = "Medium", "yellow"
                else:
                    churn_risk, risk_color = "Low", "green"

                reasons = ["Model predicted based on historical e-commerce features"]
                best_model_name = str(best_model.__class__.__name__).replace("Classifier", "")

                state_col = db['ecommerce_dataset_state']
                state = state_col.find_one({'user_id': session['user_id']})
                best_accuracy = state.get('results', {}).get('best_score', 0.68) * 100 if state else 68.0
            except Exception:
                model_path = None

        if not (model_path and scaler_path):
            reasons = []
            risk_score = 0
            if tenure_months < 6:
                risk_score += 25
                reasons.append("Low account tenure (< 6 months)")
            elif tenure_months < 12:
                risk_score += 10
                reasons.append("Medium account tenure (6-12 months)")
                
            if customer_satisfaction_score <= 4:
                risk_score += 30
                reasons.append("Very low customer satisfaction score (<= 4)")
            elif customer_satisfaction_score <= 6:
                risk_score += 15
                reasons.append("Moderate customer satisfaction score (5-6)")
                
            if days_since_last_order >= 30:
                risk_score += 20
                reasons.append("Inactive for more than 30 days")
                
            if return_rate > 0.4:
                risk_score += 15
                reasons.append("High return rate (> 40%)")
                
            if total_orders < 3:
                risk_score += 10
                reasons.append("Low purchase activity (< 3 orders)")

            risk_score = min(risk_score, 100)
            if risk_score > 60:
                churn_risk, risk_color = "High", "red"
            elif risk_score > 30:
                churn_risk, risk_color = "Medium", "yellow"
            else:
                churn_risk, risk_color = "Low", "green"
            
            if not reasons: reasons.append("Stable shopping behavior and high satisfaction")
            best_model_name = None
            best_accuracy = None

        result = {
            'churn_risk': churn_risk, 'risk_score': risk_score, 'reasons': reasons, 'risk_color': risk_color,
            'best_model_name': best_model_name, 'best_accuracy': best_accuracy
        }

        predictions_collection.insert_one({
            'user_id': session['user_id'], 'customer_id': form.get('customer_id', 'Manual Entry'),
            'industry': 'ecommerce', 'age': age, 'gender': gender, 'tenure_months': tenure_months,
            'city': city, 'total_orders': total_orders, 'average_order_value': average_order_value,
            'days_since_last_order': days_since_last_order, 'return_rate': return_rate,
            'customer_satisfaction_score': customer_satisfaction_score, 'churn_risk': churn_risk,
            'risk_score': risk_score, 'reasons': reasons, 'dataset_filename': get_dataset_filename('ecommerce'), 'created_at': datetime.now()
        })

    return render_template('ecommerce_customer_predict.html', result=result, form=form)


@app.route('/predict/ecommerce/upload', methods=['GET', 'POST'])
@login_required
def ecommerce_upload():
    import os
    results = None
    filename = None
    state_col = db['ecommerce_dataset_state']
    
    state = state_col.find_one({'user_id': session['user_id']})
    if state:
        results = state.get('results')
        filename = state.get('filename')

    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        action = request.form.get('action', 'train')

        if not csv_file or csv_file.filename == '':
            flash('No file selected. Please upload a CSV file.', 'danger')
            return render_template('ecommerce_upload.html', results=results, filename=filename)

        if not csv_file.filename.lower().endswith('.csv'):
            flash('Invalid file type. Only .csv files are accepted.', 'danger')
            return render_template('ecommerce_upload.html', results=results, filename=filename)

        try:
            csv_bytes = csv_file.stream.read()
            stream = io.StringIO(csv_bytes.decode('utf-8-sig'), newline=None)

            from ecommerce_train import train_ecommerce_models
            tune_xgboost = (action == 'tune')
            results = train_ecommerce_models(stream, tune_xgboost=tune_xgboost)
            
            os.makedirs('dump models', exist_ok=True)
            with open('dump models/ecommerce_last_dataset.csv', 'wb') as f:
                f.write(csv_bytes)
                
            state_col.update_one(
                {'user_id': session['user_id']},
                {'$set': {
                    'filename': csv_file.filename,
                    'results': results,
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
            filename = csv_file.filename
            
            flash(f'Training complete! Best model: {results["best_model"]} with accuracy {results["best_score"]*100:.2f}%.', 'success')
            return render_template('ecommerce_upload.html', results=results, filename=filename)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Error during training: {str(e)}', 'danger')
            return render_template('ecommerce_upload.html', results=results, filename=filename)

    return render_template('ecommerce_upload.html', results=results, filename=filename)


@app.route('/predict/ecommerce/visualize', methods=['GET', 'POST'])
@login_required
def ecommerce_visualize():
    import os
    csv_file = None
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')

    if not csv_file:
        local_path = 'dump models/ecommerce_last_dataset.csv'
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8-sig') as f:
                    stream = io.StringIO(f.read(), newline=None)
                from ecommerce_visualize import generate_visualizations
                visualizations, churn_rate, retention_rate = generate_visualizations(stream)
                return render_template('ecommerce_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
            except Exception as e:
                flash(f'Error reading saved dataset: {str(e)}', 'danger')
                return redirect(url_for('ecommerce_upload'))
        else:
            flash('No active dataset found to visualize. Please upload one first.', 'danger')
            return redirect(url_for('ecommerce_upload'))

    if not csv_file.filename.lower().endswith('.csv'):
        flash('Invalid file type. Only .csv files are accepted.', 'danger')
        return redirect(url_for('ecommerce_upload'))

    try:
        stream = io.StringIO(csv_file.stream.read().decode('utf-8-sig'), newline=None)
        from ecommerce_visualize import generate_visualizations
        visualizations, churn_rate, retention_rate = generate_visualizations(stream)
        return render_template('ecommerce_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
    except Exception as e:
        flash(f'Error generating visualizations: {str(e)}', 'danger')
        return redirect(url_for('ecommerce_upload'))


@app.route('/predict/ecommerce/remove_dataset', methods=['POST'])
@login_required
def ecommerce_remove_dataset():
    import os
    state_col = db['ecommerce_dataset_state']
    state_col.delete_one({'user_id': session['user_id']})
    
    local_path = 'dump models/ecommerce_last_dataset.csv'
    if os.path.exists(local_path):
        os.remove(local_path)
        
    model_path = os.path.join('dump models', 'ecommerce_best_model.pkl')
    scaler_path = os.path.join('dump models', 'ecommerce_scaler.pkl')
    features_path = os.path.join('dump models', 'ecommerce_features.pkl')
    
    for path in [model_path, scaler_path, features_path]:
        if os.path.exists(path):
            os.remove(path)
            
    flash('Active dataset and all trained models/operations have been successfully removed.', 'success')
    return redirect(url_for('ecommerce_upload'))

@app.route('/predict/subscription')
@login_required
def predict_subscription():
    return redirect(url_for('subscription_upload'))

@app.route('/predict/subscription/customer', methods=['GET', 'POST'])
@login_required
def subscription_customer_predict():
    result = None
    form   = None

    if request.method == 'POST':
        # Collect all form fields
        form = request.form.to_dict()

        def safe_float(key, default=0.0):
            try:
                return float(form.get(key, default) or default)
            except (ValueError, TypeError):
                return default

        age = safe_float('age')
        gender = form.get('gender', 'Female')
        city = form.get('city', 'Nagpur')
        tenure = safe_float('subscription_tenure_months')
        sub_type = form.get('subscription_type', 'Premium')
        monthly_fee = safe_float('monthly_subscription_fee')
        usage_hours = safe_float('usage_hours_per_week')
        days_since_last = safe_float('days_since_last_login')
        device = form.get('device_type', 'Mobile')
        failures = safe_float('payment_failure_count')
        tickets = safe_float('customer_support_tickets')
        satisfaction = safe_float('customer_satisfaction_score')

        # Attempt to load trained model
        import os, joblib
        import pandas as pd
        models_dir = 'dump models'

        model_candidates = ['subscription_best_model.pkl', 'subscription_svm_model.pkl', 'subscription_model.pkl']
        scaler_candidates = ['subscription_scaler.pkl', 'subscription_standardscaler.pkl']
        features_path = os.path.join(models_dir, 'subscription_features.pkl')

        model_path = next((os.path.join(models_dir, f) for f in model_candidates if os.path.exists(os.path.join(models_dir, f))), None)
        scaler_path = next((os.path.join(models_dir, f) for f in scaler_candidates if os.path.exists(os.path.join(models_dir, f))), None)

        if model_path and scaler_path:
            best_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            if os.path.exists(features_path):
                feature_cols = joblib.load(features_path)
            else:
                feature_cols = ['age','subscription_tenure_months','monthly_subscription_fee','usage_hours_per_week','days_since_last_login','payment_failure_count','customer_support_tickets','customer_satisfaction_score']

            input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
            if 'age' in input_data.columns: input_data['age'] = age
            if 'subscription_tenure_months' in input_data.columns: input_data['subscription_tenure_months'] = tenure
            if 'monthly_subscription_fee' in input_data.columns: input_data['monthly_subscription_fee'] = monthly_fee
            if 'usage_hours_per_week' in input_data.columns: input_data['usage_hours_per_week'] = usage_hours
            if 'days_since_last_login' in input_data.columns: input_data['days_since_last_login'] = days_since_last
            if 'payment_failure_count' in input_data.columns: input_data['payment_failure_count'] = failures
            if 'customer_support_tickets' in input_data.columns: input_data['customer_support_tickets'] = tickets
            if 'customer_satisfaction_score' in input_data.columns: input_data['customer_satisfaction_score'] = satisfaction

            g_col = f'gender_{gender}'
            if g_col in input_data.columns: input_data[g_col] = 1
            
            s_col = f'subscription_type_{sub_type}'
            if s_col in input_data.columns: input_data[s_col] = 1

            d_col = f'device_type_{device}'
            if d_col in input_data.columns: input_data[d_col] = 1

            c_col = f'city_{city}'
            if c_col in input_data.columns: input_data[c_col] = 1

            try:
                input_scaled = scaler.transform(input_data)
                prob = best_model.predict_proba(input_scaled)[0]
                churn_prob = prob[1] * 100

                risk_score = float(round(churn_prob, 2))
                if risk_score > 60:
                    churn_risk, risk_color = "High", "red"
                elif risk_score > 30:
                    churn_risk, risk_color = "Medium", "yellow"
                else:
                    churn_risk, risk_color = "Low", "green"

                reasons = ["Model predicted based on historical subscription features"]
                best_model_name = str(best_model.__class__.__name__).replace("Classifier", "")

                state_col = db['subscription_dataset_state']
                state = state_col.find_one({'user_id': session['user_id']})
                best_accuracy = state.get('results', {}).get('best_score', 0.68) * 100 if state else 68.0
            except Exception:
                model_path = None

        if not (model_path and scaler_path):
            # Fallback to rule-based engine
            reasons = []
            risk_score = 0
            if tenure < 6:
                risk_score += 25
                reasons.append("Low subscription tenure (< 6 months)")
            elif tenure < 12:
                risk_score += 10
                reasons.append("Medium subscription tenure (6-12 months)")
                
            if satisfaction <= 4:
                risk_score += 30
                reasons.append("Very low customer satisfaction score (<= 4)")
            elif satisfaction <= 6:
                risk_score += 15
                reasons.append("Moderate customer satisfaction score (5-6)")
                
            if tickets >= 4:
                risk_score += 20
                reasons.append("High number of support tickets (>= 4)")
                
            if failures >= 2:
                risk_score += 15
                reasons.append("Multiple failed payments (>= 2)")
                
            if days_since_last >= 30:
                risk_score += 15
                reasons.append("Inactive for more than 30 days")
                
            if usage_hours < 5:
                risk_score += 10
                reasons.append("Low weekly usage (< 5 hours)")

            risk_score = min(risk_score, 100)
            if risk_score > 60:
                churn_risk, risk_color = "High", "red"
            elif risk_score > 30:
                churn_risk, risk_color = "Medium", "yellow"
            else:
                churn_risk, risk_color = "Low", "green"
            
            if not reasons:
                reasons.append("Stable usage patterns and high satisfaction score")
                
            best_model_name = None
            best_accuracy = None

        result = {
            'churn_risk':  churn_risk,
            'risk_score':  risk_score,
            'reasons':     reasons,
            'risk_color':  risk_color,
            'best_model_name': best_model_name,
            'best_accuracy': best_accuracy
        }

        # Save to predictions collection
        prediction_doc = {
            'user_id':        session['user_id'],
            'customer_id':    form.get('customer_id', 'Manual Entry'),
            'industry':       'subscription',
            'tenure':         tenure,
            'monthly_charges': monthly_fee,
            'total_charges':  monthly_fee * max(1, tenure),
            'contract_type':  sub_type,
            'payment_method': device,
            'churn_risk':     churn_risk,
            'risk_score':     risk_score,
            'reasons':        reasons,
            'dataset_filename': get_dataset_filename('subscription'),
            'created_at':     datetime.now()
        }
        predictions_collection.insert_one(prediction_doc)

    return render_template('subscription_customer_predict.html', result=result, form=form)


@app.route('/predict/subscription/upload', methods=['GET', 'POST'])
@login_required
def subscription_upload():
    import os
    results = None
    filename = None
    state_col = db['subscription_dataset_state']
    
    # Load permanent state
    state = state_col.find_one({'user_id': session['user_id']})
    if state:
        results = state.get('results')
        filename = state.get('filename')

    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        action = request.form.get('action', 'train')

        # Validation
        if not csv_file or csv_file.filename == '':
            flash('No file selected. Please upload a CSV file.', 'danger')
            return render_template('subscription_upload.html', results=results, filename=filename)

        if not csv_file.filename.lower().endswith('.csv'):
            flash('Invalid file type. Only .csv files are accepted.', 'danger')
            return render_template('subscription_upload.html', results=results, filename=filename)

        try:
            # Read CSV content into stream to pass to the functions
            csv_bytes = csv_file.stream.read()
            stream = io.StringIO(csv_bytes.decode('utf-8-sig'), newline=None)

            from subscription_train import train_subscription_models
            
            # Determine if we should only tune XGBoost
            tune_xgboost = (action == 'tune')
            
            # Train models and get results
            results = train_subscription_models(stream, tune_xgboost=tune_xgboost)
            
            # Save the file locally so we can run visualizations on it later
            os.makedirs('dump models', exist_ok=True)
            with open('dump models/subscription_last_dataset.csv', 'wb') as f:
                f.write(csv_bytes)
                
            # Save state to MongoDB
            state_col.update_one(
                {'user_id': session['user_id']},
                {'$set': {
                    'filename': csv_file.filename,
                    'results': results,
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
            filename = csv_file.filename
            
            flash(f'Training complete! Best model: {results["best_model"]} with accuracy {results["best_score"]*100:.2f}%.', 'success')
            return render_template('subscription_upload.html', results=results, filename=filename)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Error during training: {str(e)}', 'danger')
            return render_template('subscription_upload.html', results=results, filename=filename)

    return render_template('subscription_upload.html', results=results, filename=filename)


@app.route('/predict/subscription/visualize', methods=['GET', 'POST'])
@login_required
def subscription_visualize():
    import os
    csv_file = None
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')

    # If GET or no file uploaded in POST, check if we have an active dataset saved locally
    if not csv_file:
        local_path = 'dump models/subscription_last_dataset.csv'
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8-sig') as f:
                    stream = io.StringIO(f.read(), newline=None)
                from subscription_visualize import generate_visualizations
                visualizations, churn_rate, retention_rate = generate_visualizations(stream)
                return render_template('subscription_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
            except Exception as e:
                flash(f'Error reading saved dataset: {str(e)}', 'danger')
                return redirect(url_for('subscription_upload'))
        else:
            flash('No active dataset found to visualize. Please upload one first.', 'danger')
            return redirect(url_for('subscription_upload'))

    # If POST with a new file:
    if not csv_file.filename.lower().endswith('.csv'):
        flash('Invalid file type. Only .csv files are accepted.', 'danger')
        return redirect(url_for('subscription_upload'))

    try:
        stream = io.StringIO(csv_file.stream.read().decode('utf-8-sig'), newline=None)
        from subscription_visualize import generate_visualizations
        visualizations, churn_rate, retention_rate = generate_visualizations(stream)
        return render_template('subscription_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
    except Exception as e:
        flash(f'Error generating visualizations: {str(e)}', 'danger')
        return redirect(url_for('subscription_upload'))


@app.route('/predict/subscription/remove_dataset', methods=['POST'])
@login_required
def subscription_remove_dataset():
    import os
    state_col = db['subscription_dataset_state']
    state_col.delete_one({'user_id': session['user_id']})
    
    # Delete locally saved dataset
    local_path = 'dump models/subscription_last_dataset.csv'
    if os.path.exists(local_path):
        os.remove(local_path)
        
    # Also delete the saved models, scaler, and features so predictions fallback to rule-based!
    model_path = os.path.join('dump models', 'subscription_best_model.pkl')
    scaler_path = os.path.join('dump models', 'subscription_scaler.pkl')
    features_path = os.path.join('dump models', 'subscription_features.pkl')
    
    for path in [model_path, scaler_path, features_path]:
        if os.path.exists(path):
            os.remove(path)
            
    flash('Active dataset and all trained models/operations have been successfully removed.', 'success')
    return redirect(url_for('subscription_upload'))

@app.route('/predict/fitness')
@login_required
def predict_fitness():
    return redirect(url_for('fitness_upload'))

@app.route('/predict/fitness/customer', methods=['GET', 'POST'])
@login_required
def fitness_customer_predict():
    result = None
    form   = None

    if request.method == 'POST':
        form = request.form.to_dict()
        def safe_float(key, default=0.0):
            try: return float(form.get(key, default) or default)
            except (ValueError, TypeError): return default

        age = safe_float('age', 28.0)
        gender = form.get('gender', 'Male')
        bmi = safe_float('bmi', 23.5)
        distance_from_gym_km = safe_float('distance_from_gym_km', 3.0)
        membership_type = form.get('membership_type', 'Monthly')
        membership_duration_months = safe_float('membership_duration_months', 12.0)
        monthly_fee = safe_float('monthly_fee', 1500.0)
        renewal_count = safe_float('renewal_count', 1.0)
        payment_delay_count = safe_float('payment_delay_count', 0.0)
        goal_type = form.get('goal_type', 'Fitness')
        attendance_frequency_per_week = safe_float('attendance_frequency_per_week', 3.0)
        last_checkin_days_ago = safe_float('last_checkin_days_ago', 5.0)
        workout_sessions_completed = safe_float('workout_sessions_completed', 40.0)
        app_usage_hours = safe_float('app_usage_hours', 2.0)
        personal_trainer = 1 if form.get('personal_trainer') == '1' else 0
        group_class_participation = 1 if form.get('group_class_participation') == '1' else 0
        upi_payment_usage = 1 if form.get('upi_payment_usage') == '1' else 0

        import os, joblib
        import pandas as pd
        models_dir = 'dump models'

        model_candidates = ['gym_best_model.pkl', 'gym_svm_model.pkl', 'gym_model.pkl']
        scaler_candidates = ['gym_scaler.pkl', 'gym_standardscaler.pkl']
        features_path = os.path.join(models_dir, 'gym_features.pkl')

        model_path = next((os.path.join(models_dir, f) for f in model_candidates if os.path.exists(os.path.join(models_dir, f))), None)
        scaler_path = next((os.path.join(models_dir, f) for f in scaler_candidates if os.path.exists(os.path.join(models_dir, f))), None)

        if model_path and scaler_path:
            best_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            if os.path.exists(features_path):
                feature_cols = joblib.load(features_path)
            else:
                feature_cols = ['age','bmi','distance_from_gym_km','membership_duration_months','monthly_fee','renewal_count','payment_delay_count','attendance_frequency_per_week','last_checkin_days_ago','workout_sessions_completed','app_usage_hours','personal_trainer','group_class_participation','upi_payment_usage']

            input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
            if 'age' in input_data.columns: input_data['age'] = age
            if 'bmi' in input_data.columns: input_data['bmi'] = bmi
            if 'distance_from_gym_km' in input_data.columns: input_data['distance_from_gym_km'] = distance_from_gym_km
            if 'membership_duration_months' in input_data.columns: input_data['membership_duration_months'] = membership_duration_months
            if 'monthly_fee' in input_data.columns: input_data['monthly_fee'] = monthly_fee
            if 'renewal_count' in input_data.columns: input_data['renewal_count'] = renewal_count
            if 'payment_delay_count' in input_data.columns: input_data['payment_delay_count'] = payment_delay_count
            if 'attendance_frequency_per_week' in input_data.columns: input_data['attendance_frequency_per_week'] = attendance_frequency_per_week
            if 'last_checkin_days_ago' in input_data.columns: input_data['last_checkin_days_ago'] = last_checkin_days_ago
            if 'workout_sessions_completed' in input_data.columns: input_data['workout_sessions_completed'] = workout_sessions_completed
            if 'app_usage_hours' in input_data.columns: input_data['app_usage_hours'] = app_usage_hours
            if 'personal_trainer' in input_data.columns: input_data['personal_trainer'] = personal_trainer
            if 'group_class_participation' in input_data.columns: input_data['group_class_participation'] = group_class_participation
            if 'upi_payment_usage' in input_data.columns: input_data['upi_payment_usage'] = upi_payment_usage

            g_col = f'gender_{gender}'
            if g_col in input_data.columns: input_data[g_col] = 1
            
            m_col = f'membership_type_{membership_type}'
            if m_col in input_data.columns: input_data[m_col] = 1
            
            goal_col = f'goal_type_{goal_type}'
            if goal_col in input_data.columns: input_data[goal_col] = 1

            try:
                input_scaled = scaler.transform(input_data)
                prob = best_model.predict_proba(input_scaled)[0]
                churn_prob = prob[1] * 100

                risk_score = float(round(churn_prob, 2))
                if risk_score > 60:
                    churn_risk, risk_color = "High", "red"
                elif risk_score > 30:
                    churn_risk, risk_color = "Medium", "yellow"
                else:
                    churn_risk, risk_color = "Low", "green"

                reasons = ["Model predicted based on historical fitness/gym features"]
                best_model_name = str(best_model.__class__.__name__).replace("Classifier", "")

                state_col = db['gym_dataset_state']
                state = state_col.find_one({'user_id': session['user_id']})
                best_accuracy = state.get('results', {}).get('best_score', 0.68) * 100 if state else 68.0
            except Exception:
                model_path = None

        if not (model_path and scaler_path):
            reasons = []
            risk_score = 0
            if last_checkin_days_ago >= 15:
                risk_score += 30
                reasons.append("Last gym visit was more than 15 days ago")
            elif last_checkin_days_ago >= 7:
                risk_score += 15
                reasons.append("Inactive in the past week")
                
            if attendance_frequency_per_week < 1.5:
                risk_score += 25
                reasons.append("Low weekly attendance frequency (< 1.5 days)")
                
            if payment_delay_count >= 2:
                risk_score += 20
                reasons.append("Multiple delayed membership payments")
                
            if personal_trainer == 0:
                risk_score += 10
                reasons.append("No active personal trainer subscription")
                
            if membership_duration_months < 6:
                risk_score += 15
                reasons.append("Short membership duration (< 6 months)")

            risk_score = min(risk_score, 100)
            if risk_score > 60:
                churn_risk, risk_color = "High", "red"
            elif risk_score > 30:
                churn_risk, risk_color = "Medium", "yellow"
            else:
                churn_risk, risk_color = "Low", "green"
            
            if not reasons: reasons.append("Frequent gym attendance and consistent membership patterns")
            best_model_name = None
            best_accuracy = None

        result = {
            'churn_risk': churn_risk, 'risk_score': risk_score, 'reasons': reasons, 'risk_color': risk_color,
            'best_model_name': best_model_name, 'best_accuracy': best_accuracy
        }

        predictions_collection.insert_one({
            'user_id': session['user_id'], 'customer_id': form.get('customer_id', 'Manual Entry'),
            'industry': 'fitness', 'age': age, 'gender': gender, 'bmi': bmi,
            'distance_from_gym_km': distance_from_gym_km, 'membership_type': membership_type,
            'membership_duration_months': membership_duration_months, 'monthly_fee': monthly_fee,
            'renewal_count': renewal_count, 'payment_delay_count': payment_delay_count,
            'goal_type': goal_type, 'attendance_frequency_per_week': attendance_frequency_per_week,
            'last_checkin_days_ago': last_checkin_days_ago, 'workout_sessions_completed': workout_sessions_completed,
            'app_usage_hours': app_usage_hours, 'personal_trainer': personal_trainer,
            'group_class_participation': group_class_participation, 'upi_payment_usage': upi_payment_usage,
            'churn_risk': churn_risk, 'risk_score': risk_score, 'reasons': reasons, 'dataset_filename': get_dataset_filename('gym'), 'created_at': datetime.now()
        })

    return render_template('fitness_customer_predict.html', result=result, form=form)


@app.route('/predict/fitness/upload', methods=['GET', 'POST'])
@login_required
def fitness_upload():
    import os
    results = None
    filename = None
    state_col = db['gym_dataset_state']
    
    state = state_col.find_one({'user_id': session['user_id']})
    if state:
        results = state.get('results')
        filename = state.get('filename')

    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        action = request.form.get('action', 'train')

        if not csv_file or csv_file.filename == '':
            flash('No file selected. Please upload a CSV file.', 'danger')
            return render_template('fitness_upload.html', results=results, filename=filename)

        if not csv_file.filename.lower().endswith('.csv'):
            flash('Invalid file type. Only .csv files are accepted.', 'danger')
            return render_template('fitness_upload.html', results=results, filename=filename)

        try:
            csv_bytes = csv_file.stream.read()
            stream = io.StringIO(csv_bytes.decode('utf-8-sig'), newline=None)

            from gym_train import train_gym_models
            tune_xgboost = (action == 'tune')
            results = train_gym_models(stream, tune_xgboost=tune_xgboost)
            
            os.makedirs('dump models', exist_ok=True)
            with open('dump models/gym_last_dataset.csv', 'wb') as f:
                f.write(csv_bytes)
                
            state_col.update_one(
                {'user_id': session['user_id']},
                {'$set': {
                    'filename': csv_file.filename,
                    'results': results,
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
            filename = csv_file.filename
            
            flash(f'Training complete! Best model: {results["best_model"]} with accuracy {results["best_score"]*100:.2f}%.', 'success')
            return render_template('fitness_upload.html', results=results, filename=filename)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Error during training: {str(e)}', 'danger')
            return render_template('fitness_upload.html', results=results, filename=filename)

    return render_template('fitness_upload.html', results=results, filename=filename)


@app.route('/predict/fitness/visualize', methods=['GET', 'POST'])
@login_required
def fitness_visualize():
    import os
    csv_file = None
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')

    if not csv_file:
        local_path = 'dump models/gym_last_dataset.csv'
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8-sig') as f:
                    stream = io.StringIO(f.read(), newline=None)
                from gym_visualize import generate_visualizations
                visualizations, churn_rate, retention_rate = generate_visualizations(stream)
                return render_template('fitness_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
            except Exception as e:
                flash(f'Error reading saved dataset: {str(e)}', 'danger')
                return redirect(url_for('fitness_upload'))
        else:
            flash('No active dataset found to visualize. Please upload one first.', 'danger')
            return redirect(url_for('fitness_upload'))

    if not csv_file.filename.lower().endswith('.csv'):
        flash('Invalid file type. Only .csv files are accepted.', 'danger')
        return redirect(url_for('fitness_upload'))

    try:
        stream = io.StringIO(csv_file.stream.read().decode('utf-8-sig'), newline=None)
        from gym_visualize import generate_visualizations
        visualizations, churn_rate, retention_rate = generate_visualizations(stream)
        return render_template('fitness_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
    except Exception as e:
        flash(f'Error generating visualizations: {str(e)}', 'danger')
        return redirect(url_for('fitness_upload'))


@app.route('/predict/fitness/remove_dataset', methods=['POST'])
@login_required
def fitness_remove_dataset():
    import os
    state_col = db['gym_dataset_state']
    state_col.delete_one({'user_id': session['user_id']})
    
    local_path = 'dump models/gym_last_dataset.csv'
    if os.path.exists(local_path):
        os.remove(local_path)
        
    model_path = os.path.join('dump models', 'gym_best_model.pkl')
    scaler_path = os.path.join('dump models', 'gym_scaler.pkl')
    features_path = os.path.join('dump models', 'gym_features.pkl')
    
    for path in [model_path, scaler_path, features_path]:
        if os.path.exists(path):
            os.remove(path)
            
    flash('Active dataset and all trained models/operations have been successfully removed.', 'success')
    return redirect(url_for('fitness_upload'))


@app.route('/predict/rideshare')
@login_required
def predict_rideshare():
    return redirect(url_for('rideshare_upload'))

@app.route('/predict/rideshare/customer', methods=['GET', 'POST'])
@login_required
def rideshare_customer_predict():
    result = None
    form   = None

    if request.method == 'POST':
        form = request.form.to_dict()
        def safe_float(key, default=0.0):
            try: return float(form.get(key, default) or default)
            except (ValueError, TypeError): return default

        age = safe_float('age', 29.0)
        gender = form.get('gender', 'Female')
        city = form.get('city', 'Nagpur')
        tenure_months = safe_float('tenure_months', 18.0)
        total_rides = safe_float('total_rides', 145.0)
        average_ride_cost = safe_float('average_ride_cost', 240.0)
        days_since_last_ride = safe_float('days_since_last_ride', 5.0)
        cancellation_rate = safe_float('cancellation_rate', 0.08)
        app_usage_hours_per_week = safe_float('app_usage_hours_per_week', 4.2)
        membership_type = form.get('membership_type', 'Basic')
        payment_method = form.get('payment_method', 'Credit Card')
        customer_satisfaction_score = safe_float('customer_satisfaction_score', 8.0)

        import os, joblib
        import pandas as pd
        models_dir = 'dump models'

        model_candidates = ['rideshare_best_model.pkl', 'rideshare_rf_model.pkl', 'rideshare_model.pkl']
        scaler_candidates = ['rideshare_scaler.pkl', 'rideshare_standardscaler.pkl']
        features_path = os.path.join(models_dir, 'rideshare_features.pkl')

        model_path = next((os.path.join(models_dir, f) for f in model_candidates if os.path.exists(os.path.join(models_dir, f))), None)
        scaler_path = next((os.path.join(models_dir, f) for f in scaler_candidates if os.path.exists(os.path.join(models_dir, f))), None)

        if model_path and scaler_path:
            best_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            if os.path.exists(features_path):
                feature_cols = joblib.load(features_path)
            else:
                feature_cols = ['age','tenure_months','total_rides','average_ride_cost','days_since_last_ride','cancellation_rate','app_usage_hours_per_week','customer_satisfaction_score']

            input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
            if 'age' in input_data.columns: input_data['age'] = age
            if 'tenure_months' in input_data.columns: input_data['tenure_months'] = tenure_months
            if 'total_rides' in input_data.columns: input_data['total_rides'] = total_rides
            if 'average_ride_cost' in input_data.columns: input_data['average_ride_cost'] = average_ride_cost
            if 'days_since_last_ride' in input_data.columns: input_data['days_since_last_ride'] = days_since_last_ride
            if 'cancellation_rate' in input_data.columns: input_data['cancellation_rate'] = cancellation_rate
            if 'app_usage_hours_per_week' in input_data.columns: input_data['app_usage_hours_per_week'] = app_usage_hours_per_week
            if 'customer_satisfaction_score' in input_data.columns: input_data['customer_satisfaction_score'] = customer_satisfaction_score

            g_col = f'gender_{gender}'
            if g_col in input_data.columns: input_data[g_col] = 1
            
            c_col = f'city_{city}'
            if c_col in input_data.columns: input_data[c_col] = 1
            
            m_col = f'membership_type_{membership_type}'
            if m_col in input_data.columns: input_data[m_col] = 1
            
            p_col = f'payment_method_{payment_method}'
            if p_col in input_data.columns: input_data[p_col] = 1

            try:
                input_scaled = scaler.transform(input_data)
                prob = best_model.predict_proba(input_scaled)[0]
                churn_prob = prob[1] * 100

                risk_score = float(round(churn_prob, 2))
                if risk_score > 60:
                    churn_risk, risk_color = "High", "red"
                elif risk_score > 30:
                    churn_risk, risk_color = "Medium", "yellow"
                else:
                    churn_risk, risk_color = "Low", "green"

                reasons = ["Model predicted based on historical rideshare features"]
                best_model_name = str(best_model.__class__.__name__).replace("Classifier", "")

                state_col = db['rideshare_dataset_state']
                state = state_col.find_one({'user_id': session['user_id']})
                best_accuracy = state.get('results', {}).get('best_score', 0.68) * 100 if state else 68.0
            except Exception:
                model_path = None

        if not (model_path and scaler_path):
            reasons = []
            risk_score = 0
            if days_since_last_ride >= 20:
                risk_score += 30
                reasons.append("Inactive for more than 20 days")
            elif days_since_last_ride >= 10:
                risk_score += 15
                reasons.append("No ride taken in the past 10 days")
                
            if cancellation_rate > 0.2:
                risk_score += 20
                reasons.append("High ride cancellation rate (> 20%)")
                
            if customer_satisfaction_score <= 4:
                risk_score += 25
                reasons.append("Very low customer satisfaction score (<= 4)")
            elif customer_satisfaction_score <= 6:
                risk_score += 10
                reasons.append("Moderate customer satisfaction score (5-6)")
                
            if tenure_months < 6:
                risk_score += 15
                reasons.append("Low driver/rider tenure (< 6 months)")

            risk_score = min(risk_score, 100)
            if risk_score > 60:
                churn_risk, risk_color = "High", "red"
            elif risk_score > 30:
                churn_risk, risk_color = "Medium", "yellow"
            else:
                churn_risk, risk_color = "Low", "green"
            
            if not reasons: reasons.append("Consistent riding activity and high satisfaction score")
            best_model_name = None
            best_accuracy = None

        result = {
            'churn_risk': churn_risk, 'risk_score': risk_score, 'reasons': reasons, 'risk_color': risk_color,
            'best_model_name': best_model_name, 'best_accuracy': best_accuracy
        }

        predictions_collection.insert_one({
            'user_id': session['user_id'], 'customer_id': form.get('customer_id', 'Manual Entry'),
            'industry': 'rideshare', 'age': age, 'gender': gender, 'city': city,
            'tenure_months': tenure_months, 'total_rides': total_rides,
            'average_ride_cost': average_ride_cost, 'days_since_last_ride': days_since_last_ride,
            'cancellation_rate': cancellation_rate, 'app_usage_hours_per_week': app_usage_hours_per_week,
            'membership_type': membership_type, 'payment_method': payment_method,
            'customer_satisfaction_score': customer_satisfaction_score, 'churn_risk': churn_risk,
            'risk_score': risk_score, 'reasons': reasons, 'created_at': datetime.now()
        })

    return render_template('rideshare_customer_predict.html', result=result, form=form)


@app.route('/predict/rideshare/upload', methods=['GET', 'POST'])
@login_required
def rideshare_upload():
    import os
    results = None
    filename = None
    state_col = db['rideshare_dataset_state']
    
    state = state_col.find_one({'user_id': session['user_id']})
    if state:
        results = state.get('results')
        filename = state.get('filename')

    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        action = request.form.get('action', 'train')

        if not csv_file or csv_file.filename == '':
            flash('No file selected. Please upload a CSV file.', 'danger')
            return render_template('rideshare_upload.html', results=results, filename=filename)

        if not csv_file.filename.lower().endswith('.csv'):
            flash('Invalid file type. Only .csv files are accepted.', 'danger')
            return render_template('rideshare_upload.html', results=results, filename=filename)

        try:
            csv_bytes = csv_file.stream.read()
            stream = io.StringIO(csv_bytes.decode('utf-8-sig'), newline=None)

            from rideshare_train import train_rideshare_models
            tune_xgboost = (action == 'tune')
            results = train_rideshare_models(stream, tune_xgboost=tune_xgboost)
            
            os.makedirs('dump models', exist_ok=True)
            with open('dump models/rideshare_last_dataset.csv', 'wb') as f:
                f.write(csv_bytes)
                
            state_col.update_one(
                {'user_id': session['user_id']},
                {'$set': {
                    'filename': csv_file.filename,
                    'results': results,
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
            filename = csv_file.filename
            
            flash(f'Training complete! Best model: {results["best_model"]} with accuracy {results["best_score"]*100:.2f}%.', 'success')
            return render_template('rideshare_upload.html', results=results, filename=filename)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Error during training: {str(e)}', 'danger')
            return render_template('rideshare_upload.html', results=results, filename=filename)

    return render_template('rideshare_upload.html', results=results, filename=filename)


@app.route('/predict/rideshare/visualize', methods=['GET', 'POST'])
@login_required
def rideshare_visualize():
    import os
    csv_file = None
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')

    if not csv_file:
        local_path = 'dump models/rideshare_last_dataset.csv'
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8-sig') as f:
                    stream = io.StringIO(f.read(), newline=None)
                from rideshare_visualize import generate_visualizations
                visualizations, churn_rate, retention_rate = generate_visualizations(stream)
                return render_template('rideshare_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
            except Exception as e:
                flash(f'Error reading saved dataset: {str(e)}', 'danger')
                return redirect(url_for('rideshare_upload'))
        else:
            flash('No active dataset found to visualize. Please upload one first.', 'danger')
            return redirect(url_for('rideshare_upload'))

    if not csv_file.filename.lower().endswith('.csv'):
        flash('Invalid file type. Only .csv files are accepted.', 'danger')
        return redirect(url_for('rideshare_upload'))

    try:
        stream = io.StringIO(csv_file.stream.read().decode('utf-8-sig'), newline=None)
        from rideshare_visualize import generate_visualizations
        visualizations, churn_rate, retention_rate = generate_visualizations(stream)
        return render_template('rideshare_visualize.html', visualizations=visualizations, churn_rate=churn_rate, retention_rate=retention_rate)
    except Exception as e:
        flash(f'Error generating visualizations: {str(e)}', 'danger')
        return redirect(url_for('rideshare_upload'))


@app.route('/predict/rideshare/remove_dataset', methods=['POST'])
@login_required
def rideshare_remove_dataset():
    import os
    state_col = db['rideshare_dataset_state']
    state_col.delete_one({'user_id': session['user_id']})
    
    local_path = 'dump models/rideshare_last_dataset.csv'
    if os.path.exists(local_path):
        os.remove(local_path)
        
    model_path = os.path.join('dump models', 'rideshare_best_model.pkl')
    scaler_path = os.path.join('dump models', 'rideshare_scaler.pkl')
    features_path = os.path.join('dump models', 'rideshare_features.pkl')
    
    for path in [model_path, scaler_path, features_path]:
        if os.path.exists(path):
            os.remove(path)
            
    flash('Active dataset and all trained models/operations have been successfully removed.', 'success')
    return redirect(url_for('rideshare_upload'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/help')
def help():
    return render_template('help.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/prediction/view')
@login_required
def view_prediction():
    customer_id = request.args.get('customer_id')
    industry = request.args.get('industry')
    if not customer_id:
        flash('Missing customer id', 'danger')
        return redirect(url_for('dashboard'))

    # find most recent prediction for this user/customer/industry
    query = {'user_id': session['user_id'], 'customer_id': customer_id}
    if industry:
        query['industry'] = industry

    pred = predictions_collection.find_one(query, sort=[('created_at', -1)])
    if not pred:
        flash('No prediction found for this customer', 'warning')
        return redirect(url_for('dashboard'))

    return render_template('prediction_detail.html', prediction=pred)


@app.route('/prediction/history')
@login_required
def prediction_history():
    industry = request.args.get('industry')
    customer_id = request.args.get('customer_id')

    # Build query scoped to the logged-in user and optional filters
    query = {'user_id': session['user_id']}
    if industry:
        query['industry'] = industry
    if customer_id:
        query['customer_id'] = customer_id

    entries = list(predictions_collection.find(query).sort('created_at', -1).limit(200))
    return render_template('prediction_history.html', entries=entries, industry=industry, customer_id=customer_id)


@app.route('/api/prediction/history')
@login_required
def api_prediction_history():
    customer_id = request.args.get('customer_id')
    industry = request.args.get('industry')
    dataset = request.args.get('dataset')
    try:
        limit = int(request.args.get('limit', 5))
    except Exception:
        limit = 5
    try:
        offset = int(request.args.get('offset', 0))
    except Exception:
        offset = 0

    query = {'user_id': session['user_id']}
    if industry:
        query['industry'] = industry
    if customer_id:
        query['customer_id'] = customer_id
    if dataset:
        query['dataset_filename'] = dataset

    cursor = predictions_collection.find(query).sort('created_at', -1).skip(offset).limit(limit)
    entries = []
    for e in cursor:
        entries.append({
            'id': str(e.get('_id')),
            'created_at': e.get('created_at').isoformat() if e.get('created_at') else None,
            'industry': e.get('industry'),
            'customer_id': e.get('customer_id'),
            'churn_risk': e.get('churn_risk'),
            'risk_score': e.get('risk_score'),
            'dataset_filename': e.get('dataset_filename')
        })

    total = predictions_collection.count_documents(query)
    return jsonify({'entries': entries, 'total': total})


@app.route('/api/datasets')
@login_required
def api_datasets():
    # Return available uploaded dataset filename for each industry for this user
    industries = ['telecom','banking','ecommerce','subscription','gym','rideshare']
    result = {}
    for ind in industries:
        try:
            col = db.get_collection(f"{ind}_dataset_state")
            state = col.find_one({'user_id': session['user_id']})
            if state and state.get('filename'):
                result[ind] = state.get('filename')
        except Exception:
            continue
    return jsonify(result)





if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


@app.errorhandler(Exception)
def handle_exception(e):
    # Log full traceback to a file
    tb = traceback.format_exc()
    logging.basicConfig(filename='error.log', level=logging.ERROR)
    logging.error(tb)
    # Render friendly error page
    return render_template('error.html'), 500
