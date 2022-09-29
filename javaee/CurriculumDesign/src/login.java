package	servlets;
import entity.*;
import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;

@WebServlet("/login")
public class login extends HttpServlet {
    public login(){
        super();
    }
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        doPost(request, response);
    }

public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("UTF-8");
        response.setContentType("text/html;charset=UTF-8");
        HttpSession session = request.getSession();
        String url="";
		if(new user(request).check()) {
			url="index.jsp";
            session.setAttribute("login","true");
		}else {
            session.setAttribute("login","false");
			url="login.jsp";
		}	
		
		request.getRequestDispatcher(url).forward(request, response);
}
}