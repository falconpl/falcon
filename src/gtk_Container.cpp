/**
 *  \file gtk_Container.cpp
 */

#include "gtk_Container.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {


void Container::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Container = mod->addClass( "Container", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Widget" ) );
    c_Container->getClassDef()->addInheritance( in );

    mod->addClassMethod( c_Container, "add",    &Container::add ).asSymbol()
        ->addParam( "widget" );

}

/*#
    @class gtk.Container
    @brief Abstract container class.

    This is the abstract container from which all gtk+ widgets which hold other
    items derive from. It mainly houses virtual functions used for inserting
    and removing children. Containers in gtk+ may hold one item or many items
    depending on the implementation.
 */

/*#
    @method add gtk.Container
    @brief Add a widget to the container.
    @param widget The widget
 */
FALCON_FUNC Container::add( VMARG )
{
    Item* i_wdt = vm->param( 0 );

    if ( !i_wdt || i_wdt->isNil() ||
        !( i_wdt->isOfClass( "Widget" ) || i_wdt->isOfClass( "gtk.Widget" ) ) )
    {
        throw_inv_params( "[Widget]" );
    }

    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    gtk_container_add( (GtkContainer*)_obj, wdt );
}


} // Gtk
} // Falcon
