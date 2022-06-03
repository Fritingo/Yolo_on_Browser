const $pokemonList = document.getElementById("pokemon-list");
const $backToTopBtn = document.getElementById("back-to-top");;
const $searchForm = document.getElementById("search-form");
const $getDuck = document.getElementById("1");
const $getPanda = document.getElementById("2");
const $getZebra = document.getElementById("3");
const $getTiger = document.getElementById("4");
const $getKoala = document.getElementById("5");
const $getLion = document.getElementById("6");
const $getFish = document.getElementById("7");
const $getPenguin = document.getElementById("8");
const $getSnail = document.getElementById("9");
const $getChicken = document.getElementById("10");
const $getRabbit = document.getElementById("11");
const $getPig = document.getElementById("12");
const $getMonkey = document.getElementById("13");
const $getDog = document.getElementById("14");
const $getCat = document.getElementById("15");

function backToTop() {
  window.scrollTo({top: 0, behavior: 'smooth'})
}


$backToTopBtn.addEventListener("click", backToTop);

$getDuck.addEventListener("click", () => {
  console.log('click');
  document.getElementById("Duck").removeAttribute("style");
});

$getPanda.addEventListener("click", () => {
  console.log('click');
  document.getElementById("Panda").removeAttribute("style");
});

$getZebra.addEventListener("click", () => {
  console.log('click');
  document.getElementById("Zebra").removeAttribute("style");
});

$getTiger.addEventListener("click", () => {
  document.getElementById("Tiger").removeAttribute("style");
});

$getKoala.addEventListener("click", () => {
  document.getElementById("Koala").removeAttribute("style");
});

$getLion.addEventListener("click", () => {
  document.getElementById("Lion").removeAttribute("style");
});

$getFish.addEventListener("click", () => {
  document.getElementById("Fish").removeAttribute("style");
});

$getPenguin.addEventListener("click", () => {
  document.getElementById("Penguin").removeAttribute("style");
});

$getSnail.addEventListener("click", () => {
  document.getElementById("Snail").removeAttribute("style");
});

$getChicken.addEventListener("click", () => {
  document.getElementById("Chicken").removeAttribute("style");
});

$getRabbit.addEventListener("click", () => {
  document.getElementById("Rabbit").removeAttribute("style");
});

$getPig.addEventListener("click", () => {
  document.getElementById("Pig").removeAttribute("style");
});

$getMonkey.addEventListener("click", () => {
  document.getElementById("Monkey").removeAttribute("style");
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








