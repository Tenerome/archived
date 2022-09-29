package	servlets;
import entity.*;
import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;

@WebServlet("/register")
public class register extends HttpServlet {
    public register(){
        super();
    }
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        doPost(request, response);
    }

public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("UTF-8");
        response.setContentType("text/html;charset=UTF-8");
        String register="";
        String url="";
		if(new user(request).register()) {
			url+="login.jsp";
		}else {
            url+="register.jsp";
			register+="false";
		}
		request.setAttribute("register", register);
		
		request.getRequestDispatcher(url).forward(request, response);
}
}