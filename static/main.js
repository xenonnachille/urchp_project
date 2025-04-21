const initialConditions = [
    "sin(pi * x)",
    "exp(-((x-1)**2)/0.1)",
    "x * (1 - x)",
    "heaviside(x - 0.5)",
    "0.5 + 0.5 * sin(2*pi*x)",
    "[0.0, 0.25, 0.5, 0.25, 0.0]",
    "abs(x - 0.5)"
];

const boundaryConditionValues = [
    "0",
    "1",
    "sin(t)",
    "cos(pi*t)",
    "0.5*u + 1",
    "x + t",
    "exp(-t)"
];

const sourceTerms = [
    "0",
    "x * cos(2*pi*t)",
    "sin(pi*x)*exp(-t)",
    "x**2 - t",
    "10 * exp(-((x-1)**2)/0.05)",
    "heaviside(x - 1)",
    "sin(5*pi*x)*cos(2*pi*t)"
];


const updateLabel = (id) => {
  document.getElementById(`val-${id}`).innerText = document.getElementById(id).value;
}

['alpha', 'length', 'nx', 'nt', 'dt'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => updateLabel(id));
});

async function solve() {
    try {
        const alpha = parseFloat(document.getElementById("alpha").value);
        const length = parseFloat(document.getElementById("length").value);
        const nx = parseInt(document.getElementById("nx").value);
        const nt = parseInt(document.getElementById("nt").value);
        const dt = parseFloat(document.getElementById("dt").value);
        const mode = document.getElementById("mode").value;

        const params = {
        alpha,
        length,
        nx,
        nt,
        dt,
        initial_condition: document.getElementById("initial_condition").value,
        boundary_conditions: {
            left: {
            type: document.getElementById("left_bc_type").value,
            value: document.getElementById("left_bc_value").value
            },
            right: {
            type: document.getElementById("right_bc_type").value,
            value: document.getElementById("right_bc_value").value
            }
        },
        source_term: document.getElementById("source_term").value,
        scheme: document.getElementById("scheme").value
        };

        const response = await fetch("/solve/heat-equation/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params)
        });

        if (!response.ok) throw new Error(await response.text());

        const data = await response.json();
        const z = data.solution;

        const x = Array.from({ length: nx }, (_, i) => i * length / (nx - 1));
        const t = Array.from({ length: nt }, (_, i) => i * dt);

        if (mode === "3d") {
        const surface = {
            type: 'surface',
            z: z,
            x: t,
            y: x
        };

        const layout = {
            title: 'Heat Equation Solution (3D)',
            autosize: true,
            scene: {
            xaxis: { title: 'Time' },
            yaxis: { title: 'Position' },
            zaxis: { title: 'Temperature' }
            }
        };

        Plotly.newPlot('plot', [surface], layout);
        } else {
            const frames = t.map((time, i) => ({
                name: i.toString(),
                data: [{
                x: x,
                y: z[i],
                mode: 'lines',
                type: 'scatter'
                }]
            }));

            const initialData = [{
                x: x,
                y: z[0],
                mode: 'lines',
                type: 'scatter'
            }];

            const layout = {
                title: 'Heat Equation Animation',
                xaxis: { title: 'Position' },
                yaxis: {
                title: 'Temperature',
                range: [Math.min(...z.flat()), Math.max(...z.flat())]
                },
                updatemenus: [{
                type: "buttons",
                showactive: false,
                x: 0.1,
                y: 1.15,
                direction: "left",
                buttons: [
                    {
                    label: "Play",
                    method: "animate",
                    args: [null, {
                        fromcurrent: true,
                        frame: { duration: 100, redraw: true },
                        transition: { duration: 0 }
                    }]
                    }
                ]
                }],
                sliders: [{
                active: 0,
                pad: { t: 30 },
                steps: t.map((_, i) => ({
                    label: `t=${(i * dt).toFixed(2)}`,
                    method: "animate",
                    args: [[i.toString()], {
                    mode: "immediate",
                    frame: { duration: 0, redraw: true },
                    transition: { duration: 0 }
                    }]
                })),
                currentvalue: {
                    visible: true,
                    prefix: "Time: ",
                    font: { size: 14, color: "#333" }
                }
                }]
            };

            const config = { responsive: true };

            Plotly.newPlot('plot', initialData, layout, config).then(() => {
                Plotly.animate('plot', null, {
                frame: { duration: 0, redraw: false },
                transition: { duration: 0 },
                mode: "immediate"
                });

                Plotly.addFrames('plot', frames);
            });
        }



    } catch (error) {
        console.error("Ошибка при решении:", error);
        document.getElementById("plot").innerHTML = `<p style="color:red;">Ошибка: ${error.message}</p>`;
    }
}