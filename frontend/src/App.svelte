<!-- App.svelte -->
<script>
	let selectedImage = null;
	// ocrResult를 배열 대신 객체로 초기화합니다.
	let ocrResult = null;
	let isLoading = false;
	let processingTime = null;

	async function handleImageUpload(event) {
		const file = event.target.files[0];
		if (file) {
			// 이미지 미리보기를 위한 URL 생성
			selectedImage = URL.createObjectURL(file);
			
			// OCR 처리를 위한 FormData 생성
			const formData = new FormData();
			formData.append('file', file);
			
			isLoading = true;
			try {
				// FastAPI 서버로 요청
				const response = await fetch('http://localhost:8000/ocr/', {
					method: 'POST',
					headers: {
						'accept': 'application/json',
						// FormData 사용 시 Content-Type은 자동 설정됨
					},
					body: formData
				});

				console.log('Response status:', response.status);

				if (!response.ok) {
					const errorText = await response.text();
					console.error('Error response:', errorText);
					throw new Error('OCR 처리 중 오류가 발생했습니다.');
				}
				
				const data = await response.json();
				console.log('OCR result:', data);
				// 기존에는 전체 응답을 저장했으나,
				// 이제는 data 객체 안의 정보를 따로 저장합니다.
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
					<p><strong>종류:</strong> {ocrResult.head}</p>
					<p><strong>이름:</strong> {ocrResult.name}</p>
					<p><strong>주민번호:</strong> {ocrResult.idnumber}</p>
					<p><strong>주소:</strong> {ocrResult.address}</p>
					<p><strong>발급일:</strong> {ocrResult.Issued_date}</p>
					<p><strong>발급기관:</strong> {ocrResult.Issuer}</p>
                    <p class="processing-time">처리 시간: {processingTime.toFixed(2)}초</p>
				</div>
			{/if}
		</div>
	</div>
</main>

<style>
	.header {
		padding: 1rem 2rem;
		text-align: center;
		margin-bottom: 2rem;
		border-bottom: 2px solid #e5e7eb;
	}

	.header h1 {
		margin: 0;
		font-size: 2rem;
		font-weight: bold;
		color: #1f2937;
	}

	.container {
		display: flex;
		gap: 2rem;
		padding: 2rem;
	}

	.left-panel, .right-panel {
		flex: 1;
		min-height: 500px;
		border: 1px solid #ccc;
		padding: 1rem;
		border-radius: 8px;
	}

	.left-panel {
		position: relative;
		display: flex;
		flex-direction: column;
		width: 500px;
		height: 700px;
	}

	.image-container {
		flex: 1;
		position: relative;
		overflow: hidden;
	}

	.image-container img {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}
</style>
