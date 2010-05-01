/**
 *  \file gtk_WindowGroup.cpp
 */

#include "gtk_WindowGroup.hpp"

#include "gtk_Window.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void WindowGroup::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_WindowGroup = mod->addClass( "GtkWindowGroup", &WindowGroup::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_WindowGroup->getClassDef()->addInheritance( in );

    c_WindowGroup->getClassDef()->factory( &WindowGroup::factory );

    Gtk::MethodTab methods[] =
    {
    { "add_window",     &WindowGroup::add_window },
    { "remove_window",  &WindowGroup::remove_window },
#if GTK_MINOR_VERSION >= 14
    { "list_windows",   &WindowGroup::list_windows },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_WindowGroup, meth->name, meth->cb );
}


WindowGroup::WindowGroup( const Falcon::CoreClass* gen, const GtkWindowGroup* wingrp )
    :
    Gtk::CoreGObject( gen, (GObject*) wingrp )
{}


Falcon::CoreObject* WindowGroup::factory( const Falcon::CoreClass* gen, void* wingrp, bool )
{
    return new WindowGroup( gen, (GtkWindowGroup*) wingrp );
}


/*#
    @class GtkWindowGroup
    @brief Limit the effect of grabs

    Creates a new GtkWindowGroup object.
    Grabs added with gtk_grab_add() only affect windows within the same GtkWindowGroup.
 */
FALCON_FUNC WindowGroup::init( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    self->setGObject( (GObject*) gtk_window_group_new() );
}


/*#
    @method add_window GtkWindowGroup
    @brief Adds a window to a GtkWindowGroup.
    @param window the GtkWindow to add.
 */
FALCON_FUNC WindowGroup::add_window( VMARG )
{
    Item* i_win = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_win || i_win->isNil() || !i_win->isObject()
        || !IS_DERIVED( i_win, GtkWindow ) )
        throw_inv_params( "GtkWindow" );
#endif
    GtkWindow* win = (GtkWindow*) COREGOBJECT( i_win )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_window_group_add_window( (GtkWindowGroup*)_obj, win );
}


/*#
    @method remove_window GtkWindowGroup
    @brief Removes a window from a GtkWindowGroup.
    @param window the GtkWindow to remove.
 */
FALCON_FUNC WindowGroup::remove_window( VMARG )
{
    Item* i_win = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_win || i_win->isNil() || !i_win->isObject()
        || !IS_DERIVED( i_win, GtkWindow ) )
        throw_inv_params( "GtkWindow" );
#endif
    GtkWindow* win = (GtkWindow*) COREGOBJECT( i_win )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_window_group_remove_window( (GtkWindowGroup*)_obj, win );
}


#if GTK_MINOR_VERSION >= 14
/*#
    @method list_windows GtkWindowGroup
    @brief Returns a list of the GtkWindows that belong to window_group.
    @return an array of GtkWindow.
 */
FALCON_FUNC WindowGroup::list_windows( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GList* lst = gtk_window_group_list_windows( (GtkWindowGroup*)_obj );
    GList* el;
    int num = 0;
    for ( el = lst; el; el = el->next, ++num );
    CoreArray* arr = new CoreArray( num );
    if ( num )
    {
        Item* wki = vm->findWKI( "GtkWindow" );
        for ( el = lst; el; el = el->next )
            arr->append( new Gtk::Window( wki->asClass(), (GtkWindow*) el->data ) );
    }
    vm->retval( arr );
}
#endif


} // Gtk
} // Falcon
