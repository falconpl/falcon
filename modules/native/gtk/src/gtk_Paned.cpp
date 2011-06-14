/**
 *  \file gtk_Paned.cpp
 */

#include "gtk_Paned.hpp"

#include "gtk_Widget.hpp"
#include "gtk_Window.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Paned::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Paned = mod->addClass( "GtkPaned", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_Paned->getClassDef()->addInheritance( in );

    c_Paned->getClassDef()->factory( &Paned::factory );

    mod->addClassMethod( c_Paned, "add1",       &Paned::add1 );
    mod->addClassMethod( c_Paned, "add2",       &Paned::add2 );
    mod->addClassMethod( c_Paned, "pack1",      &Paned::pack1 );
    mod->addClassMethod( c_Paned, "pack2",      &Paned::pack2 );
    mod->addClassMethod( c_Paned, "get_child1", &Paned::get_child1 );
    mod->addClassMethod( c_Paned, "get_child2", &Paned::get_child2 );
    mod->addClassMethod( c_Paned, "set_position",&Paned::set_position );
    mod->addClassMethod( c_Paned, "get_position",&Paned::get_position );
#if GTK_CHECK_VERSION( 2, 20, 0 )
    mod->addClassMethod( c_Paned, "get_handle_window",&Paned::get_handle_window );
#endif

}


Paned::Paned( const Falcon::CoreClass* gen, const GtkPaned* paned )
    :
    Gtk::CoreGObject( gen, (GObject*) paned )
{}


Falcon::CoreObject* Paned::factory( const Falcon::CoreClass* gen, void* paned, bool )
{
    return new Paned( gen, (GtkPaned*) paned );
}


/*#
    @class GtkPaned
    @brief Base class for widgets with two adjustable panes

    GtkPaned is the base class for widgets with two panes, arranged either horizontally
    (GtkHPaned) or vertically (GtkVPaned). Child widgets are added to the panes of
    the widget with gtk_paned_pack1() and gtk_paned_pack2(). The division beween the
    two children is set by default from the size requests of the children, but it can
    be adjusted by the user.

    A paned widget draws a separator between the two child widgets and a small handle
    that the user can drag to adjust the division. It does not draw any relief around
    the children or around the separator. (The space in which the separator is called
    the gutter.) Often, it is useful to put each child inside a GtkFrame with the
    shadow type set to GTK_SHADOW_IN so that the gutter appears as a ridge. No
    separator is drawn if one of the children is missing.

    Each child has two options that can be set, resize and shrink. If resize is
    true, then when the GtkPaned is resized, that child will expand or shrink along
    with the paned widget. If shrink is true, then when that child can be made smaller
    than its requisition by the user. Setting shrink to FALSE allows the application
    to set a minimum size. If resize is false for both children, then this is treated
    as if resize is true for both children.

    The application can set the position of the slider as if it were set by the user,
    by calling gtk_paned_set_position().
 */


/*#
    @method add1 GtkPaned
    @brief Adds a child to the top or left pane with default parameters.
    @param child (GtkWidget) the child to add

    This is equivalent to pack1(paned, child, FALSE, TRUE).
 */
FALCON_FUNC Paned::add1( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    gtk_paned_add1( (GtkPaned*)_obj, wdt );
}


/*#
    @method add2 GtkPaned
    @brief Adds a child to the bottom or right pane with default parameters.
    @param child (GtkWidget) the child to add

    This is equivalent to pack2(paned, child, TRUE, TRUE).
 */
FALCON_FUNC Paned::add2( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    gtk_paned_add2( (GtkPaned*)_obj, wdt );
}


/*#
    @method pack1 GtkPaned
    @brief Adds a child to the top or left pane.
    @param child (GtkWidget) the child to add
    @param resize (boolean) should this child expand when the paned widget is resized.
    @param shrink (boolean) can this child be made smaller than its requisition.
 */
FALCON_FUNC Paned::pack1( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkWidget,B,B" );

    CoreGObject* o_wdt = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget,B,B" );
#endif
    GtkWidget* wdt = (GtkWidget*) o_wdt->getObject();

    gboolean resize = args.getBoolean( 1 );
    gboolean shrink = args.getBoolean( 2 );

    MYSELF;
    GET_OBJ( self );
    gtk_paned_pack1( (GtkPaned*)_obj, wdt, resize, shrink );
}


/*#
    @method pack2 GtkPaned
    @brief Adds a child to the bottom or right pane.
    @param child (GtkWidget) the child to add
    @param resize (boolean) should this child expand when the paned widget is resized.
    @param shrink (boolean) can this child be made smaller than its requisition.
 */
FALCON_FUNC Paned::pack2( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkWidget,B,B" );

    CoreGObject* o_wdt = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget,B,B" );
#endif
    GtkWidget* wdt = (GtkWidget*) o_wdt->getObject();

    gboolean resize = args.getBoolean( 1 );
    gboolean shrink = args.getBoolean( 2 );

    MYSELF;
    GET_OBJ( self );
    gtk_paned_pack2( (GtkPaned*)_obj, wdt, resize, shrink );
}


/*#
    @method get_child1 GtkPaned
    @brief Obtains the first child of the paned widget.
    @return first child, or nil if it is not set.
 */
FALCON_FUNC Paned::get_child1( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_paned_get_child1( (GtkPaned*)_obj );
    if ( wdt )
    {
        Item* wki = vm->findWKI( "GtkWidget" );
        vm->retval( new Gtk::Widget( wki->asClass(), wdt ) );
    }
    else
        vm->retnil();
}


/*#
    @method get_child2 GtkPaned
    @brief Obtains the second child of the paned widget.
    @return second child, or nil if it is not set.
 */
FALCON_FUNC Paned::get_child2( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_paned_get_child2( (GtkPaned*)_obj );
    if ( wdt )
    {
        Item* wki = vm->findWKI( "GtkWidget" );
        vm->retval( new Gtk::Widget( wki->asClass(), wdt ) );
    }
    else
        vm->retnil();
}


/*#
    @method set_position GtkPaned
    @brief Sets the position of the divider between the two panes.
    @param position pixel position of divider, a negative value means that the position is unset.
 */
FALCON_FUNC Paned::set_position( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_paned_set_position( (GtkPaned*)_obj, i_pos->asInteger() );
}


/*#
    @method get_position GtkPaned
    @brief Obtains the position of the divider between the two panes.
    @return position of the divider
 */
FALCON_FUNC Paned::get_position( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_paned_get_position( (GtkPaned*)_obj ) );
}


#if GTK_CHECK_VERSION( 2, 20, 0 )
/*#
    @method get_handle_window GtkPaned
    @brief Returns the GdkWindow of the handle.
    @return (GtkWindow) the paned's handle window.

    This function is useful when handling button or motion events because it enables
    the callback to distinguish between the window of the paned, a child and the handle.
 */
FALCON_FUNC Paned::get_handle_window( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkWindow* wdt = gtk_paned_get_handle_window( (GtkPaned*)_obj );
    if ( wdt )
    { // TODO
        //Item* wki = vm->findWKI( "GtkWindow" );
        //vm->retval( new Gtk::Window( wki->asClass(), wdt ) );
    }
    else
        vm->retnil();
}
#endif

} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
