#ifndef GTK_ITEM_HPP
#define GTK_ITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Item
 */
class Item
    :
    public Gtk::CoreGObject
{
public:

    Item( const Falcon::CoreClass*, const GtkItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC signal_deselect( VMARG );

    static void on_deselect( GtkItem*, gpointer );

    static FALCON_FUNC signal_select( VMARG );

    static void on_select( GtkItem*, gpointer );

    static FALCON_FUNC signal_toggle( VMARG );

    static void on_toggle( GtkItem*, gpointer );

    static FALCON_FUNC select( VMARG );

    static FALCON_FUNC deselect( VMARG );

    static FALCON_FUNC toggle( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_ITEM_HPP
