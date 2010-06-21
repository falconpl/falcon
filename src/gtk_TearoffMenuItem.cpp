/**
 *  \file gtk_TearoffMenuItem.cpp
 */

#include "gtk_TearoffMenuItem.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TearoffMenuItem::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TearoffMenuItem = mod->addClass( "GtkTearoffMenuItem", &TearoffMenuItem::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkMenuItem" ) );
    c_TearoffMenuItem->getClassDef()->addInheritance( in );

    //c_TearoffMenuItem->setWKS( true );
    c_TearoffMenuItem->getClassDef()->factory( &TearoffMenuItem::factory );
}


TearoffMenuItem::TearoffMenuItem( const Falcon::CoreClass* gen, const GtkTearoffMenuItem* itm )
    :
    Gtk::CoreGObject( gen, (GObject*) itm )
{}


Falcon::CoreObject* TearoffMenuItem::factory( const Falcon::CoreClass* gen, void* itm, bool )
{
    return new TearoffMenuItem( gen, (GtkTearoffMenuItem*) itm );
}


/*#
    @class GtkTearoffMenuItem
    @brief A menu item used to tear off and reattach its menu

    A GtkTearoffMenuItem is a special GtkMenuItem which is used to tear off and
    reattach its menu.

    When its menu is shown normally, the GtkTearoffMenuItem is drawn as a
    dotted line indicating that the menu can be torn off. Activating it
    causes its menu to be torn off and displayed in its own window as a tearoff menu.

    When its menu is shown as a tearoff menu, the GtkTearoffMenuItem is drawn
    as a dotted line which has a left pointing arrow graphic indicating that
    the tearoff menu can be reattached. Activating it will erase the tearoff
    menu window.
 */
FALCON_FUNC TearoffMenuItem::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_tearoff_menu_item_new() );
}


} // Gtk
} // Falcon
