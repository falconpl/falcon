/**
 *  \file gtk_Item.cpp
 */

#include "gtk_Item.hpp"

#include "gtk_Buildable.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Item::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Item = mod->addClass( "GtkItem", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_Item->getClassDef()->addInheritance( in );

    c_Item->getClassDef()->factory( &Item::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_deselect",    &Item::signal_deselect },
    { "signal_select",      &Item::signal_select },
    { "signal_toggle",      &Item::signal_toggle },
    { "select",             &Item::select },
    { "deselect",           &Item::deselect },
    { "toggle",             &Item::toggle },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Item, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_Item );
}


Item::Item( const Falcon::CoreClass* gen, const GtkItem* itm )
    :
    Gtk::CoreGObject( gen, (GObject*) itm )
{}


Falcon::CoreObject* Item::factory( const Falcon::CoreClass* gen, void* itm, bool )
{
    return new Item( gen, (GtkItem*) itm );
}


/*#
    @class GtkItem
    @brief Abstract base class for GtkItemItem, GtkListItem and GtkTreeItem

    The GtkItem widget is an abstract base class for GtkItemItem, GtkListItem and GtkTreeItem.
 */


/*#
    @method signal_deselect GtkItem
    @brief Emitted when the item is deselected.
 */
FALCON_FUNC Item::signal_deselect( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "deselect", (void*) &Item::on_deselect, vm );
}


void Item::on_deselect( GtkItem* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "deselect", "on_deselect", (VMachine*)_vm );
}


/*#
    @method signal_select GtkItem
    @brief Emitted when the item is selected.
 */
FALCON_FUNC Item::signal_select( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "select", (void*) &Item::on_select, vm );
}


void Item::on_select( GtkItem* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "select", "on_select", (VMachine*)_vm );
}


/*#
    @method signal_toggle GtkItem
    @brief Emitted when the item is toggled.
 */
FALCON_FUNC Item::signal_toggle( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "toggle", (void*) &Item::on_toggle, vm );
}


void Item::on_toggle( GtkItem* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "toggle", "on_toggle", (VMachine*)_vm );
}


/*#
    @method select GtkItem
    @brief Emits the "select" signal on the given item.
 */
FALCON_FUNC Item::select( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_item_select( (GtkItem*)_obj );
}


/*#
    @method deselect GtkItem
    @brief Emits the "deselect" signal on the given item.
 */
FALCON_FUNC Item::deselect( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_item_deselect( (GtkItem*)_obj );
}


/*#
    @method toggle GtkItem
    @brief Emits the "toggle" signal on the given item.
 */
FALCON_FUNC Item::toggle( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_item_toggle( (GtkItem*)_obj );
}


} // Gtk
} // Falcon
