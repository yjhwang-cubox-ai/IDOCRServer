<!-- App.svelte -->
<script>
	let selectedImage = null;
	let ocrResult = null;
	let isLoading = false;
	let processingTime = null;

	async function handleImageUpload(event) {
		const file = event.target.files[0];
		if (file) {
			selectedImage = URL.createObjectURL(file);
			
			const formData = new FormData();
			formData.append('file', file);
			
			isLoading = true;
			try {
				const response = await fetch('http://localhost:8000/ocr/', {
					method: 'POST',
					headers: {
						'accept': 'application/json',
					},
					body: formData
				});

				if (!response.ok) {
					const errorText = await response.text();
					throw new Error('OCR 처리 중 오류가 발생했습니다.');
				}
				
				const data = await response.json();
				ocrResult = data.data;
				processingTime = data.processing_time;
			} catch (error) {
				console.error('Error:', error);
				alert(error.message);
			} finally {
				isLoading = false;
			}
		}
	}
</script>

<main>
	<header class="header">
		<h1>OCR Server</h1>
	</header>

	<div class="container">
		<!-- 왼쪽 패널: 이미지 업로드 및 미리보기 -->
		<div class="left-panel">
			<input 
				type="file" 
				accept="image/*" 
				on:change={handleImageUpload}
			/>
			{#if selectedImage}
				<div class="image-container">
					<img src={selectedImage} alt="업로드된 이미지" />
				</div>
			{/if}
		</div>

		<!-- 오른쪽 패널: OCR 결과 -->
		<div class="right-panel">
			{#if isLoading}
				<p>OCR 처리 중...</p>
			{:else if ocrResult}
				<div class="ocr-results">
					<p class="processing-time">처리 시간: {processingTime.toFixed(2)}초</p>
					<p><strong>종류:</strong> {ocrResult.head}</p>
					<p><strong>이름:</strong> {ocrResult.name}</p>
					<p><strong>주민번호:</strong> {ocrResult.idnumber}</p>
					<p><strong>주소:</strong> {ocrResult.address}</p>
					<p><strong>발급일:</strong> {ocrResult.Issued_date}</p>
					<p><strong>발급자:</strong> {ocrResult.Issuer}</p>
				</div>
			{/if}
		</div>
	</div>
</main>

<style>
	/* Google Fonts에서 Roboto 폰트 불러오기 */
	@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

	:global(body) {
		margin: 0;
		font-family: 'Roboto', sans-serif;
		background: linear-gradient(135deg, #f6f9fc, #e9eff5);
		color: #333;
	}

	main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 2rem;
	}

	.header {
		padding: 1rem 2rem;
		text-align: center;
		margin-bottom: 2rem;
		border-bottom: 2px solid #e5e7eb;
	}

	.header h1 {
		margin: 0;
		font-size: 2.5rem;
		font-weight: 700;
		color: #2c3e50;
	}

	.container {
		display: flex;
		gap: 2rem;
	}

	.left-panel, .right-panel {
		flex: 1;
		min-height: 500px;
		padding: 1.5rem;
		border-radius: 12px;
		background-color: #ffffff;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
	}

	/* 왼쪽 패널 스타일 */
	.left-panel {
		display: flex;
		flex-direction: column;
		align-items: center;
		width: 500px;
		height: 700px;
	}

	.left-panel input[type="file"] {
		margin-bottom: 1rem;
		padding: 0.5rem 1rem;
		font-size: 1rem;
		border: 1px solid #ccc;
		border-radius: 6px;
		cursor: pointer;
		transition: border-color 0.3s;
	}
	
	.left-panel input[type="file"]:hover {
		border-color: #2980b9;
	}

	.image-container {
		flex: 1;
		width: 100%;
		position: relative;
		overflow: hidden;
		border-radius: 12px;
		border: 2px solid #e0e0e0;
	}

	.image-container img {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		object-fit: cover;
		transition: transform 0.3s ease;
	}

	.image-container img:hover {
		transform: scale(1.05);
	}

	/* 오른쪽 패널 스타일 */
	.ocr-results {
		line-height: 1.6;
	}

	.ocr-results p {
		margin: 0.5rem 0;
	}

	.ocr-results p strong {
		color: #2c3e50;
	}

	.processing-time {
		font-size: 1.1rem;
		font-weight: 600;
		color: #2980b9;
		margin-bottom: 1rem;
	}
</style>
