<%@ page contentType="text/html; charset=UTF-8" %>
<meta charset="UTF-8">
<link rel="stylesheet" type="text/css" href="./css/main.css">
<script src="./js/content.js"></script>
<script src="./js/particles.js" ></script>
<title>
专业维护
</title>
<div id="particles-js">
<canvas class="particles-js-canvas-el"></canvas>
<div class="frame">
	<br><br>
	<h2>ADDMAJOR</h2>
<div id="enter">
	<form  name='entermajor' action="addmajor" method="post">
		<div>
			<input type="text" id="mname" name="mname" placeholder="专业">
		</div>
		<div>
			<input type="text" id="college" name="college" placeholder="学院">
		</div>
			
		<div class="addbt" style="border:0;">
			<input type="submit" value="添加" />
			<input type="reset" value="取消" />
		</div>
	</form>
</div>
</div>
</div> 