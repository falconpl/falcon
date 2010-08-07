#ifndef GTK_MENUTOOLBUTTON_HPP
#define GTK_MENUTOOLBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::MenuToolButton
 */
class MenuToolButton
    :
    public Gtk::CoreGObject
{
public:

    MenuToolButton( const Falcon::CoreClass*, const GtkMenuToolButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_show_menu( VMARG );

    static void on_show_menu( GtkMenuToolButton*, gpointer );

    static FALCON_FUNC new_from_stock( VMARG );

    static FALCON_FUNC set_menu( VMARG );

    static FALCON_FUNC get_menu( VMARG );

    //static FALCON_FUNC set_arrow_tooltip( VMARG );

    static FALCON_FUNC set_arrow_tooltip_text( VMARG );

    static FALCON_FUNC set_arrow_tooltip_markup( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_MENUTOOLBUTTON_HPP
