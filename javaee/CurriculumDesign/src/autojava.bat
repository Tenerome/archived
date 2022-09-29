cd "c:\DATA\repository\code\javaee\CurriculumDesign\WEB-INF\classes\"
rd /s /q entity
rd /s /q servlets
cd "c:\DATA\repository\code\javaee\CurriculumDesign\src\"
javac -d ./ student.java
javac -d ./ db.java
javac -d ./ user.java
javac -d ./ major.java
javac -d ./ add.java
javac -d ./ addmajor.java
javac -d ./ delete.java
javac -d ./ find.java
javac -d ./ login.java
javac -d ./ update.java
javac -d ./ register.java
move entity ../WEB-INF/classes/
move servlets ../WEB-INF/classes/
