#ifndef GTK_TOOLITEM_HPP
#define GTK_TOOLITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ToolItem
 */
class ToolItem
    :
    public Gtk::CoreGObject
{
public:

    ToolItem( const Falcon::CoreClass*, const GtkToolItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

#if 0 // todo
    static FALCON_FUNC signal_create_menu_proxy( VMARG );

    static void on_create_menu_proxy( GtkToolItem*, gpointer );

    static FALCON_FUNC signal_set_tooltip( VMARG );

    static FALCON_FUNC signal_toolbar_reconfigured( VMARG );
#endif

    static FALCON_FUNC set_homogeneous( VMARG );

    static FALCON_FUNC get_homogeneous( VMARG );

    static FALCON_FUNC set_expand( VMARG );

    static FALCON_FUNC get_expand( VMARG );

    //static FALCON_FUNC set_tooltip( VMARG );

    static FALCON_FUNC set_tooltip_text( VMARG );

    static FALCON_FUNC set_tooltip_markup( VMARG );

    static FALCON_FUNC set_use_drag_window( VMARG );

    static FALCON_FUNC get_use_drag_window( VMARG );

    static FALCON_FUNC set_visible_horizontal( VMARG );

    static FALCON_FUNC get_visible_horizontal( VMARG );

    static FALCON_FUNC set_visible_vertical( VMARG );

    static FALCON_FUNC get_visible_vertical( VMARG );

    static FALCON_FUNC set_is_important( VMARG );

    static FALCON_FUNC get_is_important( VMARG );

#if GTK_MINOR_VERSION >= 20
    static FALCON_FUNC get_ellipsize_mode( VMARG );
#endif

    static FALCON_FUNC get_icon_size( VMARG );

    static FALCON_FUNC get_orientation( VMARG );

    static FALCON_FUNC get_toolbar_style( VMARG );

    static FALCON_FUNC get_relief_style( VMARG );

#if GTK_MINOR_VERSION >= 20
    static FALCON_FUNC get_text_alignment( VMARG );

    static FALCON_FUNC get_text_orientation( VMARG );
#endif

    //static FALCON_FUNC retrieve_proxy_menu_item( VMARG );

    //static FALCON_FUNC get_proxy_menu_item( VMARG ); TODO

    static FALCON_FUNC set_proxy_menu_item( VMARG );

    static FALCON_FUNC rebuild_menu( VMARG );

#if GTK_MINOR_VERSION >= 14
    static FALCON_FUNC toolbar_reconfigured( VMARG );
#endif

    //static FALCON_FUNC get_text_size_group( VMARG ); TODO

};


} // Gtk
} // Falcon

#endif // !GTK_TOOLITEM_HPP
