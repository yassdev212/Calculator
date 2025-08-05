// Mock data for initial posts
let posts = [
    {
        id: 1,
        user: {
            name: 'John Doe',
            profilePic: 'https://via.placeholder.com/40'
        },
        content: 'Just had an amazing weekend! 🎉',
        timestamp: '2 hours ago',
        likes: 15,
        comments: [
            { user: 'Jane Smith', content: 'Looks awesome!' },
            { user: 'Mike Johnson', content: 'Great times!' }
        ],
        isLiked: false
    },
    {
        id: 2,
        user: {
            name: 'Jane Smith',
            profilePic: 'https://via.placeholder.com/40'
        },
        content: 'Check out this beautiful sunset! 🌅',
        image: 'https://via.placeholder.com/600x400',
        timestamp: '5 hours ago',
        likes: 32,
        comments: [
            { user: 'John Doe', content: 'Beautiful view!' }
        ],
        isLiked: false
    }
];

// Function to create a new post element
function createPostElement(post) {
    const postElement = document.createElement('div');
    postElement.className = 'post';
    postElement.innerHTML = `
        <div class="post-header">
            <img src="${post.user.profilePic}" alt="${post.user.name}" class="profile-img">
            <div>
                <strong>${post.user.name}</strong>
                <div class="timestamp">${post.timestamp}</div>
            </div>
        </div>
        <div class="post-content">
            <p>${post.content}</p>
            ${post.image ? `<img src="${post.image}" alt="Post image" class="post-image">` : ''}
        </div>
        <div class="post-stats">
            <span>${post.likes} likes</span> · 
            <span>${post.comments.length} comments</span>
        </div>
        <div class="post-actions">
            <button class="action-button ${post.isLiked ? 'liked' : ''}" onclick="toggleLike(${post.id})">
                <i class="like-icon"></i> Like
            </button>
            <button class="action-button" onclick="toggleComments(${post.id})">
                <i class="comment-icon"></i> Comment
            </button>
            <button class="action-button">
                <i class="share-icon"></i> Share
            </button>
        </div>
        <div class="comments-section" id="comments-${post.id}" style="display: none;">
            ${post.comments.map(comment => `
                <div class="comment">
                    <strong>${comment.user}</strong> ${comment.content}
                </div>
            `).join('')}
            <div class="add-comment">
                <img src="https://via.placeholder.com/40" alt="Profile" class="profile-img">
                <input type="text" placeholder="Write a comment..." onkeypress="handleCommentSubmit(event, ${post.id})">
            </div>
        </div>
    `;
    return postElement;
}

// Function to toggle like on a post
function toggleLike(postId) {
    const post = posts.find(p => p.id === postId);
    if (post) {
        post.isLiked = !post.isLiked;
        post.likes += post.isLiked ? 1 : -1;
        renderPosts();
    }
}

// Function to toggle comments section
function toggleComments(postId) {
    const commentsSection = document.getElementById(`comments-${postId}`);
    if (commentsSection) {
        commentsSection.style.display = commentsSection.style.display === 'none' ? 'block' : 'none';
    }
}

// Function to handle comment submission
function handleCommentSubmit(event, postId) {
    if (event.key === 'Enter' && event.target.value.trim()) {
        const post = posts.find(p => p.id === postId);
        if (post) {
            post.comments.push({
                user: 'John Doe',
                content: event.target.value.trim()
            });
            event.target.value = '';
            renderPosts();
        }
    }
}

let selectedImage = null;

// Function to handle image upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            selectedImage = e.target.result;
            document.getElementById('image-preview').src = selectedImage;
            document.getElementById('image-preview-container').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

// Function to remove image preview
function removeImagePreview() {
    selectedImage = null;
    document.getElementById('image-preview').src = '';
    document.getElementById('image-preview-container').style.display = 'none';
    document.getElementById('image-upload').value = '';
}

// Function to create a new post
function createNewPost(content) {
    if (!content.trim() && !selectedImage) return;
    
    const newPost = {
        id: posts.length + 1,
        user: {
            name: 'John Doe',
            profilePic: 'https://via.placeholder.com/40'
        },
        content: content,
        image: selectedImage,
        timestamp: 'Just now',
        likes: 0,
        comments: [],
        isLiked: false
    };
    posts.unshift(newPost);
    selectedImage = null;
    document.getElementById('image-preview-container').style.display = 'none';
    document.getElementById('image-preview').src = '';
    document.getElementById('image-upload').value = '';
    renderPosts();
}

// Function to render all posts
function renderPosts() {
    const postsContainer = document.querySelector('.posts-container');
    postsContainer.innerHTML = '';
    posts.forEach(post => {
        postsContainer.appendChild(createPostElement(post));
    });
}

// Initialize the post input functionality
function initializePostInput() {
    const postInput = document.querySelector('.post-input');
    postInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && event.target.value.trim()) {
            createNewPost(event.target.value.trim());
            event.target.value = '';
        }
    });
}

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    renderPosts();
    initializePostInput();
    
    // Initialize image upload functionality
    document.getElementById('image-upload').addEventListener('change', handleImageUpload);
    
    // Add animation classes to posts when they appear
    const postsContainer = document.querySelector('.posts-container');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });
    
    // Observe posts for animation
    document.querySelectorAll('.post').forEach(post => {
        post.style.opacity = '0';
        post.style.transform = 'translateY(20px)';
        post.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        observer.observe(post);
    });
});