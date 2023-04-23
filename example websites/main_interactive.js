function unescapeString(str) {
    return str.replace(/\\n/g, '').replace(/\\\"/g, '"');
}

async function submitForm(event) {
    event.preventDefault();

    const data = {
        principle_components: parseInt(document.getElementById('principle_components').value),
        random_state: parseInt(document.getElementById('random_state').value),
        alpha: parseFloat(document.getElementById('alpha').value),
        approx_min_span_tree: document.getElementById('approx_min_span_tree').checked,
        gen_min_span_tree: document.getElementById('gen_min_span_tree').checked,
        leaf_size: parseInt(document.getElementById('leaf_size').value),
        cluster_selection_epsilon: parseFloat(document.getElementById('cluster_selection_epsilon').value),
        metric: document.getElementById('metric').value,
        min_cluster_size: parseInt(document.getElementById('min_cluster_size').value),
        allow_single_cluster: document.getElementById('allow_single_cluster').checked
    };

    const response = await fetch('http://localhost:8080/customhdbscan', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    const jsonResponse = await response.json();

    if (jsonResponse.message === 'found') {
        const unescapedHtml = unescapeString(jsonResponse.plot);
        console.log("Unescaped HTML:", unescapedHtml);//debugging

        // Create a temporary container to hold the unescaped HTML
        const tempContainer = document.createElement('div');
        tempContainer.innerHTML = unescapedHtml;

        // Extract the script content
        const scriptTags = tempContainer.getElementsByTagName('script');

        // Insert non-script content into the 'response' div
        for (const childNode of tempContainer.childNodes) {
            if (childNode.tagName !== 'SCRIPT') {
            document.getElementById('response').appendChild(childNode.cloneNode(true));
            }
        }

        // Create new script elements and insert them into the DOM
        for (const scriptTag of scriptTags) {
            const newScript = document.createElement('script');
            newScript.innerHTML = scriptTag.innerHTML;
            document.body.appendChild(newScript);
        }





        
    } else if (jsonResponse.message === 'please wait') {
        document.getElementById('response').textContent = `Please wait ${jsonResponse.wait_time} seconds.`;
    } else {
        document.getElementById('response').textContent = 'An error occurred.';
    }
};





async function submitOtherForm(event) {
    event.preventDefault();

    const data = {
        principle_components: parseInt(document.getElementById('principle_components').value),
        random_state: parseInt(document.getElementById('random_state').value),
        alpha: parseFloat(document.getElementById('alpha').value),
        approx_min_span_tree: document.getElementById('approx_min_span_tree').checked,
        gen_min_span_tree: document.getElementById('gen_min_span_tree').checked,
        leaf_size: parseInt(document.getElementById('leaf_size').value),
        cluster_selection_epsilon: parseFloat(document.getElementById('cluster_selection_epsilon').value),
        metric: document.getElementById('metric').value,
        min_cluster_size: parseInt(document.getElementById('min_cluster_size').value),
        allow_single_cluster: document.getElementById('allow_single_cluster').checked
    };

    const response = await fetch('http://localhost:8080/cHdbscanGrp', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    const jsonResponse = await response.json();

    if (jsonResponse.message === 'found') {
        const base64Image = jsonResponse.image;
        document.getElementById('response').innerHTML = `<img src="data:image/png;base64,${base64Image}" alt="HDBSCAN Result">`;
    } else if (jsonResponse.message === 'please wait') {
        document.getElementById('response').textContent = `Please wait ${jsonResponse.wait_time} seconds.`;
    } else {
        document.getElementById('response').textContent = 'An error occurred.';
    }
};

document.getElementById("hdbscan-form").addEventListener("submit", submitForm);
document.getElementById("submit2").addEventListener("click", submitOtherForm);