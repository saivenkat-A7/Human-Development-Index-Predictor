// Quick-fill example buttons for the three scenarios described in the project brief
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("hdiForm");
  if (!form) return;

  const examples = {
    fillVeryHigh: { life_expectancy: 82.5, mean_years_schooling: 12.8, expected_years_schooling: 17.5, gni_per_capita: 55000 },
    fillMedium:   { life_expectancy: 66.0, mean_years_schooling: 6.5,  expected_years_schooling: 10.0, gni_per_capita: 6500 },
    fillLow:      { life_expectancy: 52.0, mean_years_schooling: 2.5,  expected_years_schooling: 6.0,  gni_per_capita: 1200 },
  };

  Object.keys(examples).forEach((btnId) => {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.addEventListener("click", () => {
      const values = examples[btnId];
      Object.entries(values).forEach(([name, value]) => {
        const input = form.querySelector(`[name="${name}"]`);
        if (input) input.value = value;
      });
    });
  });
});
