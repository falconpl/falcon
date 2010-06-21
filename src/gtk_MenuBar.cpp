/**
 *  \file gtk_MenuBar.cpp
 */

#include "gtk_MenuBar.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void MenuBar::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_MenuBar = mod->addClass( "GtkMenuBar", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkMenuShell" ) );
    c_MenuBar->getClassDef()->addInheritance( in );

    c_MenuBar->getClassDef()->factory( &MenuBar::factory );

    Gtk::MethodTab methods[] =
    {
#if 0 // deprecated
    { "append",     &MenuBar:: },
    { "prepend",    &MenuBar:: },
    { "insert",     &MenuBar:: },
#endif
    { "set_pack_direction",         &MenuBar::set_pack_direction },
    { "get_pack_direction",         &MenuBar::get_pack_direction },
    { "set_child_pack_direction",   &MenuBar::set_child_pack_direction },
    { "get_child_pack_direction",   &MenuBar::get_child_pack_direction },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_MenuBar, meth->name, meth->cb );
}


MenuBar::MenuBar( const Falcon::CoreClass* gen, const GtkMenuBar* bar )
    :
    Gtk::CoreGObject( gen, (GObject*) bar )
{}


Falcon::CoreObject* MenuBar::factory( const Falcon::CoreClass* gen, void* bar, bool )
{
    return new MenuBar( gen, (GtkMenuBar*) bar );
}


/*#
    @class GtkMenuBar
    @brief A subclass widget for GtkMenuShell which holds GtkMenuItem widgets

    The GtkMenuBar is a subclass of GtkMenuShell which contains one to many
    GtkMenuItem. The result is a standard menu bar which can hold many menu
    items. GtkMenuBar allows for a shadow type to be set for aesthetic
    purposes. The shadow types are defined in the set_shadow_type function.
 */
FALCON_FUNC MenuBar::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_menu_bar_new() );
}


#if 0
FALCON_FUNC MenuBar::append( VMARG );
FALCON_FUNC MenuBar::prepend( VMARG );
FALCON_FUNC MenuBar::insert( VMARG );
#endif


/*#
    @method
    @brief Sets how items should be packed inside a menubar.
    @param pack_dir a new GtkPackDirection
 */
FALCON_FUNC MenuBar::set_pack_direction( VMARG )
{
    Item* i_pack = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pack || i_pack->isNil() || !i_pack->isInteger() )
        throw_inv_params( "GtkPackDirection" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_bar_set_pack_direction( (GtkMenuBar*)_obj,
                                     (GtkPackDirection) i_pack->asInteger() );
}


/*#
    @method get_pack_direction
    @brief Retrieves the current pack direction of the menubar.
    @return the pack direction
 */
FALCON_FUNC MenuBar::get_pack_direction( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_menu_bar_get_pack_direction( (GtkMenuBar*)_obj ) );
}


/*#
    @method set_child_pack_direction
    @brief Sets how widgets should be packed inside the children of a menubar.
    @param child_pack_dir a new GtkPackDirection
 */
FALCON_FUNC MenuBar::set_child_pack_direction( VMARG )
{
    Item* i_pack = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pack || i_pack->isNil() || !i_pack->isInteger() )
        throw_inv_params( "GtkPackDirection" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_bar_set_child_pack_direction( (GtkMenuBar*)_obj,
                                           (GtkPackDirection) i_pack->asInteger() );
}


/*#
    @method get_child_pack_direction
    @brief Retrieves the current child pack direction of the menubar.
    @return the child pack direction
 */
FALCON_FUNC MenuBar::get_child_pack_direction( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_menu_bar_get_child_pack_direction( (GtkMenuBar*)_obj ) );
}


} // Gtk
} // Falcon
