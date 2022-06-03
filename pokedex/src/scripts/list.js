const $pokemonList = document.getElementById("pokemon-list");
const $backToTopBtn = document.getElementById("back-to-top");;
const $searchForm = document.getElementById("search-form");
const $getPanda = document.getElementById("2");
const $getKoala = document.getElementById("5");
const $getPenguin = document.getElementById("8");
const $getRabbit = document.getElementById("11");
const $getDog = document.getElementById("14");
const $getCat = document.getElementById("15");

function backToTop() {
  window.scrollTo({top: 0, behavior: 'smooth'})
}


$backToTopBtn.addEventListener("click", backToTop);

$getPanda.addEventListener("click", () => {
  document.getElementById("Panda").removeAttribute("style");
});

$getKoala.addEventListener("click", () => {
  document.getElementById("Koala").removeAttribute("style");
});

$getPenguin.addEventListener("click", () => {
  document.getElementById("Penguin").removeAttribute("style");
});

$getRabbit.addEventListener("click", () => {
  document.getElementById("Rabbit").removeAttribute("style");
});

$getDog.addEventListener("click", () => {
  document.getElementById("dog").removeAttribute("style");
  document.getElementById("dog").href = "dog_detail.html";
});

$getCat.addEventListener("click", () => {
  document.getElementById("cat").removeAttribute("style");
  document.getElementById("cat").href = "cat_detail.html";
});


window.addEventListener("scroll", () => {
  $backToTopBtn.style.opacity = 1;
  $backToTopBtn.style.bottom = '50px';
  
  if(window.scrollY === 0) {
    $backToTopBtn.style.opacity = 0;
    $backToTopBtn.style.bottom = '-50px';
  }
});

$searchForm.addEventListener('submit', (e) => {
  e.preventDefault();

  let pokeName = e.target[0].value;

  location.href = `details.html?name=${pokeName.toLowerCase()}`;
});








