// Toggle dropdown visibility
function toggleDropdown() {
  var dropdown = document.querySelector('.dropdown-menu');
  if (dropdown) {
    dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
  }
}

// Limit input to 2 digits
function limitDigits(input) {
  // Ensure the input is numeric and limit its length
  if (input.value.length > 2) {
    input.value = input.value.slice(0, 2);
  }
}

// Function to show loading animation and remove it after 3 seconds
document.addEventListener("DOMContentLoaded", function() {
  var loadingContainer = document.getElementById('loadingContainer');
  
  // Only proceed if the loading container exists
  if (loadingContainer) {
    // Show the loading animation
    loadingContainer.style.display = 'block';
    
    // Automatically remove the loading animation after 3 seconds
    setTimeout(function() {
      loadingContainer.style.display = 'none';
    }, 3000); // 3 seconds
  }
});
