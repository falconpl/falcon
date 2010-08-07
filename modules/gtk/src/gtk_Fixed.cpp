/**
 *  \file gtk_Fixed.cpp
 */

#include "gtk_Fixed.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Fixed::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Fixed = mod->addClass( "GtkFixed", &Fixed::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_Fixed->getClassDef()->addInheritance( in );

    c_Fixed->getClassDef()->factory( &Fixed::factory );

    mod->addClassMethod( c_Fixed, "put",            &Fixed::put );
    mod->addClassMethod( c_Fixed, "move",           &Fixed::move );
    mod->addClassMethod( c_Fixed, "get_has_window", &Fixed::get_has_window );
    mod->addClassMethod( c_Fixed, "set_has_window", &Fixed::set_has_window );

}


Fixed::Fixed( const Falcon::CoreClass* gen, const GtkFixed* fxd )
    :
    Gtk::CoreGObject( gen, (GObject*) fxd )
{}


Falcon::CoreObject* Fixed::factory( const Falcon::CoreClass* gen, void* fxd, bool )
{
    return new Fixed( gen, (GtkFixed*) fxd );
}


/*#
    @class GtkFixed
    @brief A container which allows you to position widgets at fixed coordinates

    The GtkFixed widget is a container which can place child widgets at fixed
    positions and with fixed sizes, given in pixels. GtkFixed performs no automatic
    layout management.

    For most applications, you should not use this container! It keeps you from
    having to learn about the other GTK+ containers, but it results in broken applications.
    With GtkFixed, the following things will result in truncated text, overlapping
    widgets, and other display bugs:

    * Themes, which may change widget sizes.
    * Fonts other than the one you used to write the app will of course change the size of widgets containing text; keep in mind that users may use a larger font because of difficulty reading the default, or they may be using Windows or the framebuffer port of GTK+, where different fonts are available.
    * Translation of text into other languages changes its size. Also, display of non-English text will use a different font in many cases.

    In addition, the fixed widget can't properly be mirrored in right-to-left
    languages such as Hebrew and Arabic. i.e. normally GTK+ will flip the interface
    to put labels to the right of the thing they label, but it can't do that with
    GtkFixed. So your application will not be usable in right-to-left languages.

    Finally, fixed positioning makes it kind of annoying to add/remove GUI elements,
    since you have to reposition all the other elements. This is a long-term maintenance
    problem for your application.

    If you know none of these things are an issue for your application, and prefer
    the simplicity of GtkFixed, by all means use the widget. But you should be aware
    of the tradeoffs.
 */
FALCON_FUNC Fixed::init( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* fixed = gtk_fixed_new();
    self->setObject( (GObject*) fixed );
}


/*#
    @method put GtkFixed
    @brief Adds a widget to a GtkFixed container at the given position.
    @param x the horizontal position to place the widget at.
    @param y the vertical position to place the widget at.
 */
FALCON_FUNC Fixed::put( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkWidget,I,I" );

    CoreGObject* o_wdt = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget,I,I" );
#endif
    GtkWidget* wdt = (GtkWidget*) o_wdt->getObject();

    gint x = args.getInteger( 1 );
    gint y = args.getInteger( 2 );

    MYSELF;
    GET_OBJ( self );
    gtk_fixed_put( (GtkFixed*)_obj, wdt, x, y );
}


/*#
    @method move GtkFixed
    @brief Moves a child of a GtkFixed container to the given position.
    @param x the horizontal position to move the widget to.
    @param y the vertical position to move the widget to.
 */
FALCON_FUNC Fixed::move( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkWidget,I,I" );

    CoreGObject* o_wdt = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget,I,I" );
#endif
    GtkWidget* wdt = (GtkWidget*) o_wdt->getObject();

    gint x = args.getInteger( 1 );
    gint y = args.getInteger( 2 );

    MYSELF;
    GET_OBJ( self );
    gtk_fixed_move( (GtkFixed*)_obj, wdt, x, y );
}


/*#
    @method get_has_window GtkFixed
    @brief Gets whether the GtkFixed has its own GdkWindow.
    @return true if fixed has its own window.
 */
FALCON_FUNC Fixed::get_has_window( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_fixed_get_has_window( (GtkFixed*)_obj ) );
}


/*#
    @method set_has_window GtkFixed
    @brief Sets whether a GtkFixed widget is created with a separate GdkWindow for widget->window or not.
    @param has_window true if a separate window should be created

    (By default, it will be created with no separate GdkWindow).
    This function must be called while the GtkFixed is not realized, for instance,
    immediately after the window is created.

    This function was added to provide an easy migration path for older applications
    which may expect GtkFixed to have a separate window.
 */
FALCON_FUNC Fixed::set_has_window( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_fixed_set_has_window( (GtkFixed*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


} // Gtk
} // Falcon
