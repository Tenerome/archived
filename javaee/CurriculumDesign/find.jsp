<%@ page language="java" contentType="text/html;chatset=UTF-8" pageEncoding="UTF-8"%>
<!doctype html>
<html>
<head>
<meta charset="UTF-8">
<link rel="stylesheet" type="text/css" href="./css/main.css">
<script src="./js/content.js"></script>
<script src="./js/particles.js" ></script>
<title>
查询
</title>
</head>
<body>
<div id="particles-js">
<canvas class="particles-js-canvas-el"></canvas>
<div class="frame">
	<br><br>
	<h2>QUERY</h2>
<div id="enter">
	<form name='enterstudent' action="find" method="post">
		<div>
			<input type="text" id="sno" name="sno" placeholder="学号">
		</div>
		<div>
			<input type="text" id="sname" name="sname" placeholder="姓名">
		</div>

		<div>
			<input type="text" id="smajor" name="smajor" placeholder="专业"/>			    
		</div>
		<div>
			<input type="text" id="sgrade" name="sgrade" maxlength="4" placeholder="成绩">	
		</div>	
		<div id="selector">
			<input class="selector" type="radio" id="ssex" name="ssex" value="男" style="margin-left:20px;"><label>男</label>
			
			<input class="selector" type="radio" id="ssex" name="ssex" value="女" style="margin-left:200px;"><label>女</label>
		</div>
		<div class="addbt" style="border:0;">
			<input type="submit" value="查询"/>
			<input type="reset" value="取消"/>
		</div>
	</form>
</div>
</div>
</div> 
</body>
</html>
