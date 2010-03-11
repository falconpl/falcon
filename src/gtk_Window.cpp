/**
 *  \file gtk_Window.cpp
 */

#include "gtk_Window.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Window::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Window = mod->addClass( "Window", &Window::init )
        ->addParam( "type" );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Bin" ) );
    c_Window->getClassDef()->addInheritance( in );

    mod->addClassMethod( c_Window, "get_title",     &Window::get_title );
    mod->addClassMethod( c_Window, "set_title",     &Window::set_title ).asSymbol()
        ->addParam( "title" );

}

/*#
    @class gtk.Window
    @optparam type gtk.WINDOW_TOPLEVEL (default) or gtk.WINDOW_POPUP
    @raise ParamError Invalid window type

    @prop title Window title
 */

/*#
    @init gtk.Window
 */
FALCON_FUNC Window::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
        return;

    Item* i_wtype = vm->param( 0 );

    if ( i_wtype && ( i_wtype->isNil() || !i_wtype->isInteger() ) )
    {
        throw_inv_params( "I" );
    }

    GtkWindowType gwt = GTK_WINDOW_TOPLEVEL;

    if ( i_wtype )
    {
        const int wtype = i_wtype->asInteger();

        switch ( wtype )
        {
        case GTK_WINDOW_TOPLEVEL:
            break;
        case GTK_WINDOW_POPUP:
            gwt = GTK_WINDOW_POPUP;
            break;
        default:
            throw_inv_params( FAL_STR( gtk_e_inv_window_type_ ) );
        }
    }

    GtkWidget* win = gtk_window_new( gwt );
    Gtk::internal_add_slot( (GObject*) win );
    self->setUserData( new GData( (GObject*) win ) );
}


/*#
    @method get_title gtk.Window
    @brief Get window title
 */
FALCON_FUNC Window::get_title( VMARG )
{
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
    MYSELF;
    GET_OBJ( self );
    const gchar* t = gtk_window_get_title( ((GtkWindow*)_obj) );
    vm->retval( t ? new String( t ) : new String() );
}


/*#
    @method set_title gtk.Window
    @brief Set window title
    @param title Window title
 */
FALCON_FUNC Window::set_title( VMARG )
{
    Item* it = vm->param( 0 );
    if ( !it || it->isNil() || !it->isString() )
    {
        throw_inv_params( "S" );
    }
    MYSELF;
    GET_OBJ( self );
    AutoCString s( it->asString() );
    gtk_window_set_title( ((GtkWindow*)_obj), s.c_str() );
}


} // Gtk
} // Falcon
