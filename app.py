import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load data
@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv", low_memory=False)
    ratings = pd.read_csv("Ratings.csv", low_memory=False)
    ratings = ratings.merge(books[['ISBN', 'Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']],
                            on="ISBN", how="left")
    return books, ratings

books, ratings = load_data()

# Filter data
@st.cache_data
def prepare_data(ratings):
    popular_books = ratings['Book-Title'].value_counts()
    popular_books = popular_books[popular_books > 50].index
    filtered_ratings = ratings[ratings['Book-Title'].isin(popular_books)]

    active_users = filtered_ratings['User-ID'].value_counts()
    active_users = active_users[active_users > 50].index
    filtered_ratings = filtered_ratings[filtered_ratings['User-ID'].isin(active_users)]

    return filtered_ratings

ratings = prepare_data(ratings)

# Create model
@st.cache_data
def create_model(ratings):
    book_pivot = ratings.pivot_table(
        index='Book-Title', columns='User-ID', values='Book-Rating'
    ).fillna(0)

    book_pivot_sparse = csr_matrix(book_pivot.values)
    model = NearestNeighbors(algorithm='brute', metric='cosine', n_jobs=-1)
    model.fit(book_pivot_sparse)

    return book_pivot, model

book_pivot, model = create_model(ratings)

# CSS Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
    background-attachment: fixed;
}

/* Black styling for selectbox */
.stSelectbox > div > div {
    background-color: rgb(0, 0, 0) !important;
    color: white !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 10px !important;
}

/* Main container */
.main .block-container {
    background: rgba(0, 0, 0, 0.85);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

/* Title */
h1 {
    background: linear-gradient(135deg, #ff6a00 0%, #ee0979 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 3rem !important;
    font-weight: bold;
}

/* Book card */
.book-card {
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}

.book-card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.4);
}

.book-title {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #ff6a00 0%, #ee0979 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.book-info {
    color: #ccc;
    font-size: 14px;
    margin: 4px 0;
}

.book-cover {
    width: 80px;
    height: 120px;
    object-fit: cover;
    border-radius: 8px;
    margin-right: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.book-container {
    display: flex;
    align-items: flex-start;
}

.book-details {
    flex: 1;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.5rem 2rem !important;
    font-weight: bold !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_cover_url(isbn):
    if pd.isna(isbn) or str(isbn).strip() == "":
        return "https://via.placeholder.com/100x150?text=No+Cover"
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"

def display_book_card(title, author, publisher, year, isbn, tag="Recommended"):
    author = author if pd.notna(author) else "Unknown"
    publisher = publisher if pd.notna(publisher) else "Unknown"
    year = str(int(year)) if pd.notna(year) and str(year).replace('.', '').isdigit() else "Unknown"
    cover_url = get_cover_url(isbn)

    html = f"""
    <div class="book-card">
        <div class="book-container">
            <img class="book-cover" src="{cover_url}" alt="Book cover"
                 onerror="this.src='https://via.placeholder.com/100x150?text=No+Cover'">
            <div class="book-details">
                <div class="book-title">{title}</div>
                <div class="book-info"><strong>Author:</strong> {author}</div>
                <div class="book-info"><strong>Publisher:</strong> {publisher}</div>
                <div class="book-info"><strong>Year:</strong> {year}</div>
                <div class="book-info" style="color: #0066cc;"><strong>{tag}</strong></div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def recommend_books(book_name):
    if book_name not in book_pivot.index:
        st.error("Book not found in database!")
        return

    book_id = np.where(book_pivot.index == book_name)[0][0]
    query_vector = book_pivot.iloc[book_id, :].values.reshape(1, -1)
    distances, suggestions = model.kneighbors(query_vector, n_neighbors=6)

    # Display selected book
    st.subheader("Selected Book")
    selected_book = books[books['Book-Title'] == book_name].iloc[0]
    display_book_card(
        book_name,
        selected_book['Book-Author'],
        selected_book['Publisher'],
        selected_book['Year-Of-Publication'],
        selected_book['ISBN'],
        "Your Selection"
    )

    # Display recommendations
    st.subheader("Recommendations")
    for i in range(1, len(suggestions[0])):
        recommended_title = book_pivot.index[suggestions[0][i]]
        book_details = books[books['Book-Title'] == recommended_title].iloc[0]
        display_book_card(
            recommended_title,
            book_details['Book-Author'],
            book_details['Publisher'],
            book_details['Year-Of-Publication'],
            book_details['ISBN']
        )

# Main UI
st.title("📚 FRIDAY: Book Recommender")
st.write("Select a book and get personalized recommendations")

book_list = sorted(list(book_pivot.index))
selected_book = st.selectbox("Choose a book:", book_list)

if st.button("Get Recommendations"):
    with st.spinner("Finding recommendations..."):
        recommend_books(selected_book)