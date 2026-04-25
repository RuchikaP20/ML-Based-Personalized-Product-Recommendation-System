from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__, template_folder="templates")

# load files===========================================================================================================
import pandas as pd

trending_products = pd.read_csv("C:/Users/asish/Desktop/major/E-Commerece-Recommendation-System-Machine-Learning-Product-Recommendation-system-/models/trending_products.csv")
train_data = pd.read_csv("C:/Users/asish/Desktop/major/E-Commerece-Recommendation-System-Machine-Learning-Product-Recommendation-system-/models/clean_data.csv")

# database configuration---------------------------------------
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


# def content_based_recommendations(train_data, item_name, top_n=10):
#     # Check if the item name exists in the training data
#     # if item_name not in train_data['Name'].values:
#     #     print(f"Item '{item_name}' not found in the training data.")
#     #     return pd.DataFrame()

#     # # Create a TF-IDF vectorizer for item descriptions
#     # tfidf_vectorizer = TfidfVectorizer(stop_words='english')

#     # # Apply TF-IDF vectorization to item descriptions
#     # tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

#     # # Calculate cosine similarity between items based on descriptions
#     # cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

#     # # Find the index of the item
#     # item_index = train_data[train_data['Name'] == item_name].index[0]

#     # # Get the cosine similarity scores for the item
#     # similar_items = list(enumerate(cosine_similarities_content[item_index]))

#     # # Sort similar items by similarity score in descending order
#     # similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

#     # # Get the top N most similar items (excluding the item itself)
#     # top_similar_items = similar_items[1:top_n+1]

#     # # Get the indices of the top similar items
#     # recommended_item_indices = [x[0] for x in top_similar_items]

#     # # Get the details of the top similar items
#     # recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

#     # return recommended_items_details

#     print(f"Looking for item: {item_name}")
#     train_data['Name_Lower'] = train_data['Name'].str.lower()
#     item_name_lower = item_name.lower()

#     if item_name_lower not in train_data['Name_Lower'].values:
#         print(f"Item '{item_name}' not found in the training data.")
#         return pd.DataFrame()  # Return empty dataframe when no match found

#     # Create a TF-IDF vectorizer for item descriptions
#     tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
#     cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

#     item_index = train_data[train_data['Name_Lower'] == item_name_lower].index[0]
#     similar_items = list(enumerate(cosine_similarities_content[item_index]))

#     # Sort by similarity score and select top N items
#     similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
#     top_similar_items = similar_items[1:top_n+1]  # Exclude the item itself

#     # Get the indices and details of recommended items
#     recommended_item_indices = [x[0] for x in top_similar_items]
#     recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
    
#     return recommended_items_details

from fuzzywuzzy import process

def find_closest_match(product_name, product_list):
    match, score = process.extractOne(product_name, product_list)
    return match if score > 80 else None  # Adjust threshold as needed

def content_based_recommendations(train_data, item_name, top_n=10):
    print(f"Looking for item: {item_name}")
    closest_match = find_closest_match(item_name, train_data['Name'].tolist())

    if not closest_match:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()  # Return empty dataframe when no match found

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    item_index = train_data[train_data['Name'] == closest_match].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort by similarity score and select top N items
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]  # Exclude the item itself

    # Get the indices and details of recommended items
    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
    
    return recommended_items_details
# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


@app.route("/")
def index():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

# @app.route("/main")
# def main():
#     # Now pass the variable to the template
#     return render_template('main.html', content_based_recommendations())


@app.route("/main", methods=["GET", "POST"])
def main():
    # return render_template('main.html')
    content_based_rec = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'])  # Default empty dataframe
    message = None  # Default message is None

    if request.method == "POST":
        item_name = request.form.get("prod_name")  # Get the product name
        top_n = int(request.form.get("nbr", 10))  # Default to top 10 recommendations

        if item_name:
            content_based_rec = content_based_recommendations(train_data, item_name, top_n)
            if content_based_rec.empty:
                message = "No recommendations available for this product."

    return render_template("main.html", content_based_rec=content_based_rec, message=message, truncate=truncate)

    # content_based_rec = None  # Default to None

    # if request.method == "POST":
    #     item_name = request.form.get("prod_name")  # Get the product name
    #     top_n = int(request.form.get("nbr", 10))  # Default to top 10 recommendations

    #     if item_name:
    #         # Generate recommendations
    #         content_based_rec = content_based_recommendations(train_data, item_name, top_n)

    #     # If recommendations are empty, set it to an empty DataFrame
    # #     if content_based_rec.empty:
    # #         content_based_rec = None
    # #         message = "No recommendations available for this product."
    # #         return render_template("main.html", message=message, content_based_rec=content_based_rec)

    # # return render_template("main.html", content_based_rec=content_based_rec, truncate=truncate)
    # if content_based_rec is None:
    #     content_based_rec = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'])

    # return render_template("main.html", content_based_rec = content_based_rec, truncate=truncate)

# routes
@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!'
                               )

# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username,password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed in successfully!'
                               )
# @app.route("/recommendations", methods=['POST', 'GET'])
# def recommendations():
#     if request.method == 'POST':
#         prod = request.form.get('prod')
#         nbr = int(request.form.get('nbr'))
#         content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

#         if content_based_rec.empty:
#             message = "No recommendations available for this product."
#             return render_template('main.html', message=message)
#         else:
#             # Create a list of random image URLs for each recommended product
#             random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
#             print(content_based_rec)
#             print(random_product_image_urls)

#             price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
#             return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
#                                    random_product_image_urls=random_product_image_urls,
#                                    random_price=random.choice(price))

# @app.route("/recommendations", methods=["POST", "GET"])
# def recommendations():
#     content_based_rec = None
#     if request.method == "POST":
#         prod = request.form.get('prod')  # Get the product name from the form
#         nbr = int(request.form.get('nbr'))  # Get the number of recommendations

#         # Call the content-based recommendation function
#         content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

#         if content_based_rec is None or content_based_rec.empty:
#             message = "No recommendations available for this product."
#             return render_template('main.html', message=message)
#         else:
#             # Generate random image URLs and prices
#             random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
#             price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
#             return render_template('main.html', content_based_rec=content_based_rec, 
#                                    random_product_image_urls=random_product_image_urls,
#                                    random_price=random.choice(price))

# @app.route("/recommendations", methods=['POST', 'GET'])
# def recommendations():
#     if request.method == 'POST':
#         prod = request.form.get('prod')
#         nbr = int(request.form.get('nbr'))
#         content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

#         if content_based_rec.empty:
#             message = "No recommendations available for this product."
#             return render_template('main.html', message=message)
#         else:
#             # Create a list of random image URLs for each recommended product
#             random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
#             print(content_based_rec)
#             print(random_product_image_urls)

#             price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
#             return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
#                                    random_product_image_urls=random_product_image_urls,
#                                    random_price=random.choice(price))


# @app.route("/recommendations", methods=['POST', 'GET'])
# def recommendations():
#     content_based_rec = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'])  # Default empty dataframe
    
#     if request.method == 'POST':
#         prod = request.form.get('prod')
#         nbr = int(request.form.get('nbr'))
#         content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

#         if content_based_rec.empty:
#             message = "No recommendations available for this product."
#             return render_template('main.html', message=message, content_based_rec=content_based_rec)
#         else:
#             # Generate random image URLs and prices
#             random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
#             price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
#             return render_template('main.html', content_based_rec=content_based_rec, 
#                                    random_product_image_urls=random_product_image_urls,
#                                    random_price=random.choice(price), truncate=truncate)

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    content_based_rec = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'])  # Default empty dataframe
    
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message, content_based_rec=content_based_rec)
        else:
            # Generate random image URLs and prices
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, 
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price), truncate=truncate)

if __name__=='__main__':
    app.run(debug=True)