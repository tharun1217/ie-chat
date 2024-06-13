document.addEventListener('DOMContentLoaded', function () {
    const categoryLinks = document.querySelectorAll('.category-link');
    const emailsContainer = document.getElementById('emails-container');

    categoryLinks.forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault();
            const category = this.getAttribute('data-category');

            axios.get(`/category/${category}`)
                .then(response => {
                    const emails = response.data;
                    emailsContainer.innerHTML = ''; // Clear previous emails
                    emails.forEach((email, index) => {
                        const emailDiv = document.createElement('div');
                        emailDiv.classList.add('email');
                        emailDiv.innerHTML = `
                            <div><strong>Mail #${index + 1}</strong></div>
                            <div><strong>Subject:</strong> ${email.Subject}</div>
                            <div><strong>Body:</strong> ${email.Body}</div>
                            <hr>
                        `;
                        emailsContainer.appendChild(emailDiv);
                    });
                })
                .catch(error => {
                    console.error('There was an error!', error);
                });
        });
    });
});
