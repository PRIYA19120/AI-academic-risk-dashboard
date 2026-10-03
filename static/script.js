const navItems = document.querySelectorAll(".nav-item");

navItems.forEach(item => {
    item.addEventListener("click", function(event) {

        // Sirf "#" wale links ko rokna hai
        if (this.getAttribute("href") === "#") {
            event.preventDefault();

            navItems.forEach(nav => {
                nav.classList.remove("active");
            });

            this.classList.add("active");
        }

    });
});