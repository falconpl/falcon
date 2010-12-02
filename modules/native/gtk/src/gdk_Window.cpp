/**
 *  \file gdk_Window.cpp
 */

#include "gdk_Window.hpp"

/*#
   @beginmodule gtk
*/


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Window::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Window = mod->addClass( "GdkWindow" );//, &Window::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GdkDrawable" ) );
    c_Window->getClassDef()->addInheritance( in );

    c_Window->setWKS( true );
    c_Window->getClassDef()->factory( &Window::factory );

    Gtk::MethodTab methods[] =
    {

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Window, meth->name, meth->cb );
}


Window::Window( const Falcon::CoreClass* gen, const GdkWindow* win )
    :
    Gtk::CoreGObject( gen, (GObject*) win )
{}


Falcon::CoreObject* Window::factory( const Falcon::CoreClass* gen, void* win, bool )
{
    return new Window( gen, (GdkWindow*) win );
}


/*#
    @class GdkWindow
    @brief Onscreen display areas in the target window system

    A GdkWindow is a rectangular region on the screen. It's a low-level object,
    used to implement high-level objects such as GtkWidget and GtkWindow on the
    GTK+ level. A GtkWindow is a toplevel window, the thing a user might think
    of as a "window" with a titlebar and so on; a GtkWindow may contain many
    GdkWindow. For example, each GtkButton has a GdkWindow associated with it.

    [...]
 */
//FALCON_FUNC Window::init( VMARG );


} // Gdk
} // Falcon
