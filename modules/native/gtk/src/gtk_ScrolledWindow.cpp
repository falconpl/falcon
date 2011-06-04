/**
 * \file gtk_ScrolledWindow.cpp
 */


#include "gtk_ScrolledWindow.hpp"

#include "gtk_Adjustment.hpp"

#include "gtk_Widget.hpp"

/*#
 * @beginmodule gtk
 */

namespace Falcon {
    
namespace Gtk {
        
void ScrolledWindow::modInit( Falcon::Module *mod ) {
            
    Falcon::Symbol *c_ScrolledWindow = mod->addClass( "GtkScrolledWindow", &ScrolledWindow::init );
            
    Falcon::InheritDef *in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
            
    c_ScrolledWindow->getClassDef()->addInheritance( in );
            
    c_ScrolledWindow->setWKS( true ); // We need this ? ..
            
    c_ScrolledWindow->getClassDef()->factory( &ScrolledWindow::factory );
            
    Gtk::MethodTab methods[] = {
                
        { "set_policy",         &ScrolledWindow::set_policy },
        { "get_policy",         &ScrolledWindow::get_policy },
        { "add_with_viewport",  &ScrolledWindow::add_with_viewport },
        { "set_placement",      &ScrolledWindow::set_placement },
        { "get_placement",      &ScrolledWindow::get_placement },
        { "set_hadjustment",    &ScrolledWindow::set_hadjustment },
        { "get_hadjustment",    &ScrolledWindow::get_hadjustment },
        { "set_vadjustment",    &ScrolledWindow::set_vadjustment },
        { "get_vadjustment",    &ScrolledWindow::get_vadjustment },
      //{ "get_hscrollbar",     &ScrolledWindow::get_hscrollbar },
      //{ "get_vscrollbar",     &ScrolledWindow::get_vscrollbar },
      //{ "unset_placement",    &ScrolledWindow::unset_placement },
      //{ "set_shadow_type",    &ScrolledWindow::set_shadow_type },
      //{ "get_shadow_type",    &ScrolledWindow::get_shadow_type },
        { NULL,                 NULL }
    };
            
    for( Gtk::MethodTab *currentMethod = methods; currentMethod->name; ++currentMethod ) {
                
        mod->addClassMethod( c_ScrolledWindow, currentMethod->name, currentMethod->cb );
                
    }
            
}
        
        
ScrolledWindow::ScrolledWindow( const Falcon::CoreClass *gen, const GtkScrolledWindow *wnd ) 
  : Gtk::CoreGObject( gen, (GObject*) wnd ) 
{ }
        
        
Falcon::CoreObject *ScrolledWindow::factory( const Falcon::CoreClass *gen, void *wnd, bool ) {
            
    return new ScrolledWindow( gen, (GtkScrolledWindow*) wnd );
            
}
        
/*#
 
 @class GtkScrolledWindow
 @brief Adds scrollbars to its child widget
 @param hadjustment the GtkAdjustment to use for horizontal scrollbar, or NULL to create a new adjustment
 @param vadjustment the GtkAdjustment to use for vertical scrollbar, or NULL to create a new adjustment.
 
 GtkScrolledWindow is a GtkBin subclass: it's a container the accepts a single child widget. 
 GtkScrolledWindow adds scrollbars to the child widget and optionally draws a beveled frame around the child widget.
 
 The scrolled window can work in two ways. Some widgets have native scrolling support; 
 these widgets have "slots" for GtkAdjustment objects.
 Widgets with native scroll support include GtkTreeView, GtkTextView, and GtkLayout.
 
 For widgets that lack native scrolling support,
 the GtkViewport widget acts as an adaptor class, implementing scrollability for child widgets 
 that lack their own scrolling capabilities. Use GtkViewport to scroll child widgets such as GtkTable, GtkBox, and so on.
 
 If a widget has native scrolling abilities, it can be added to the GtkScrolledWindow with gtk_container_add(). 
 If a widget does not, you must first add the widget to a GtkViewport, then add the GtkViewport to the scrolled window.
 The convenience function gtk_scrolled_window_add_with_viewport() does exactly this,
 so you can ignore the presence of the viewport.
 
 The position of the scrollbars is controlled by the scroll adjustments. 
 See GtkAdjustment for the fields in an adjustment - for GtkScrollbar, used by GtkScrolledWindow,
 the "value" field represents the position of the scrollbar, which must be between the "lower" field and "upper - page_size."
 The "page_size" field represents the size of the visible scrollable area.
 The "step_increment" and "page_increment" fields are used when the user asks to step down (using the small stepper arrows) 
 or page down (using for example the PageDown key).
 
 If a GtkScrolledWindow doesn't behave quite as you would like, or doesn't have exactly the right layout,
 it's very possible to set up your own scrolling with GtkScrollbar and for example a GtkTable.
 
 */
FALCON_FUNC ScrolledWindow::init( VMARG ) {
    
    Falcon::Item *i_h_adj = vm->param( 0 );
    Falcon::Item *i_v_adj = vm->param( 1 );
    
#ifndef NO_PARAMETER_CHECK

    if( !i_h_adj || !( i_h_adj->isNil() || ( i_h_adj->isObject() && IS_DERIVED( i_h_adj, GtkAdjustment ) ) ) || 
        !i_v_adj || !( i_v_adj->isNil() || ( i_v_adj->isObject() && IS_DERIVED( i_v_adj, GtkAdjustment ) ) ) )
            throw_inv_params( "[GtkAdjustment],[GtkAdjustment]" );
    
#endif // !NO_PARAMETER_CHECK
    
    GtkAdjustment *horizontal = i_h_adj->isNil() ? NULL : GET_ADJUSTMENT( *i_h_adj );
    GtkAdjustment *vertical   = i_v_adj->isNil() ? NULL : GET_ADJUSTMENT( *i_v_adj );

    MYSELF;

    self->setObject( (GObject*) gtk_scrolled_window_new( horizontal, vertical ) );

}
        
        
/*#
 
 @method set_policy GtkScrolledWindow
 @brief Sets the scrollbar policy for the horizontal and vertical scrollbars.
 @param hscrollbar_policy policy for horizontal bar
 @param vscrollbar_policy policy for vertical bar
 
 */
FALCON_FUNC ScrolledWindow::set_policy( VMARG ) {
    
    Item *i_hsb_policy = vm->param( 0 );
    Item *i_vsb_policy = vm->param( 1 );
    
#ifndef NO_PARAMETER_CHECK
    
    if( !i_hsb_policy || !i_hsb_policy->isInteger() || !i_vsb_policy || !i_vsb_policy->isInteger() )
        throw_inv_params("I,I");
    
    
#endif // !NO_PARAMETER_CHECK
    
    MYSELF;
    
    GET_OBJ( self );
    
    gtk_scrolled_window_set_policy( (GtkScrolledWindow*)_obj,
                                    (GtkPolicyType)i_hsb_policy->asInteger(),
                                    (GtkPolicyType)i_vsb_policy->asInteger() );

}

        
/*#

@method get_policy GtkScrolledWindow
@brief Retrieves the current policy values for the horizontal and vertical scrollbars.
@return [GtkPolicyType hsb, GtkPolicyType vsb]

*/
FALCON_FUNC ScrolledWindow::get_policy( VMARG ) {

    NO_ARGS;

    MYSELF;

    GtkPolicyType hsb_policy,vsb_policy;

    GET_OBJ( self );

    gtk_scrolled_window_get_policy( (GtkScrolledWindow*)_obj, &hsb_policy, &vsb_policy );

    CoreArray *result = new CoreArray( 2 );

    result->append( hsb_policy );
    result->append( vsb_policy );

    vm->retval( result );

}


/*#

@method add_with_viewport GtkScrolledWindow
@brief Used to add children without native scrolling capabilities. 
@param child the widget you want to scroll

*/
FALCON_FUNC ScrolledWindow::add_with_viewport( VMARG ) {

    Item *i_child = vm->param( 0 );

#ifndef NO_PARAMETER_CHECK

    if( !i_child || !( i_child->isObject() && IS_DERIVED( i_child, GtkWidget ) ) )
        throw_inv_params("GtkWidget");

#endif // !NO_PARAMETER_CHECK

    MYSELF;

    GET_OBJ( self );

    GtkWidget *child = GET_WIDGET( *i_child );

    gtk_scrolled_window_add_with_viewport( (GtkScrolledWindow*)_obj, child );

}


/*#

@method get_placement GtkScrolledWindow
@brief Gets the placement of the contents with respect to the scrollbars for the scrolled window. 
@return the current placement value

*/
FALCON_FUNC ScrolledWindow::get_placement( VMARG ) {

    NO_ARGS;

    MYSELF;

    GET_OBJ( self );

    GtkCornerType result = gtk_scrolled_window_get_placement( (GtkScrolledWindow*)_obj );

    vm->retval(result);

}


/*#

@method set_placement GtkScrolledWindow
@brief Sets the placement of the contents with respect to the scrollbars for the scrolled window. 
@param window_placement position of the child window

*/
FALCON_FUNC ScrolledWindow::set_placement( VMARG ) {

    Item *i_window_placement = vm->param( 0 );

#ifndef NO_PARAMETER_CHECK

    if( !i_window_placement || !i_window_placement->isInteger() )
        throw_inv_params("I");

#endif // !NO_PARAMETER_CHECK

    MYSELF;
    GET_OBJ( self );

    gtk_scrolled_window_set_placement( (GtkScrolledWindow*)_obj, (GtkCornerType)i_window_placement->asInteger() );

}

/*#

@method get_vadjustment GtkScrolledWindow
@brief Returns the vertical scrollbar's adjustment, 
used to connect the vertical scrollbar to the child widget's vertical scroll functionality. 
@return the vertical GtkAdjustment.

*/
FALCON_FUNC ScrolledWindow::get_vadjustment( VMARG ) {

    NO_ARGS;

    MYSELF;
    GET_OBJ( self );

    GtkAdjustment *result = gtk_scrolled_window_get_vadjustment( (GtkScrolledWindow*)_obj );

    vm->retval( new Gtk::Adjustment( vm->findWKI("GtkAdjustment")->asClass(), result ) );

}


/*#

@method set_vadjustment GtkScrolledWindow
@brief Sets the GtkAdjustment for the vertical scrollbar. 
@param vadjustment vertical scroll adjustment

*/
FALCON_FUNC ScrolledWindow::set_vadjustment( VMARG ) {

    Item *i_vadjustment = vm->param( 0 );

#ifndef NO_PARAMETER_CHECK

    if( !i_vadjustment || !( i_vadjustment->isClass() && IS_DERIVED( i_vadjustment, GtkAdjustment ) ) )
        throw_inv_params( "GtkAdjustment" );

#endif // !NO_PARAMETER_CHECK

    GtkAdjustment *adj = GET_ADJUSTMENT( *i_vadjustment );

    MYSELF;
    GET_OBJ( self );

    gtk_scrolled_window_set_vadjustment((GtkScrolledWindow*)_obj, adj);

}


/*#

@method get_hadjustment GtkScrolledWindow
@brief Returns the horizontal scrollbar's adjustment, 
used to connect the horizontal scrollbar to the child widget's horizontal scroll functionality. 
@return the vertical GtkAdjustment.

*/
FALCON_FUNC ScrolledWindow::get_hadjustment( VMARG ) {

    NO_ARGS;

    MYSELF;
    GET_OBJ( self );

    GtkAdjustment *res = gtk_scrolled_window_get_hadjustment( (GtkScrolledWindow*)_obj );

    vm->retval( new Gtk::Adjustment( vm->findWKI("GtkAdjustment")->asClass(),res ) );

}


/*#

@method set_hadjustment GtkScrolledWindow
@brief Sets the GtkAdjustment for the horizontal scrollbar. 
@param hadjustment horizontal scroll adjustment

*/
FALCON_FUNC ScrolledWindow::set_hadjustment( VMARG ) {

    Item *i_hadjustment = vm->param( 0 );

#ifndef NO_PARAMETER_CHECK

    if( !i_hadjustment || !( i_hadjustment->isClass() && IS_DERIVED( i_hadjustment, GtkAdjustment ) ) )
        throw_inv_params("GtkAdjustment");

#endif // !NO_PARAMETER_CHECK

    GtkAdjustment *adj = GET_ADJUSTMENT( *i_hadjustment );

    MYSELF;
    GET_OBJ( self );

    gtk_scrolled_window_set_hadjustment( (GtkScrolledWindow*)_obj, adj );

}

} //Gtk

} //Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;