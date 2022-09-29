<%@ page contentType="text/html; charset=UTF-8"%>
<head>
   <meta charset="UTF-8">
<link rel="stylesheet" type="text/css" href="./css/main.css">
<script src="./js/content.js"></script>
<script src="./js/particles.js" ></script>
<title>
结果
</title>
</head>
<div id="particles-js">
<canvas class="particles-js-canvas-el"></canvas>
<div class="frame">
	<br><br>

<%=request.getAttribute("info")%>
</div>
</div> 
<script type="text/javascript">
	window.onload=function(){
		var info="<%=request.getAttribute("info") %>";
		if(info!=null)
			alert(info);
		<a href="index.jsp"></a>
	}
</script>