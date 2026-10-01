const navItems = document.querySelectorAll(".nav-item");

navItems.forEach(item => {
    item.addEventListener("click", function (event) {
        event.preventDefault();

        navItems.forEach(nav => {
            nav.classList.remove("active");
        });

        this.classList.add("active");

        const pageName = this.querySelector("span:last-child").textContent;

        console.log("Selected:", pageName);
    });
});