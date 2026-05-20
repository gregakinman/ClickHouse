#include <Processors/QueryPlan/Optimizations/Optimizations.h>
#include <Processors/QueryPlan/ExpressionStep.h>
#include <Processors/QueryPlan/FilterStep.h>
#include <Processors/QueryPlan/LimitByStep.h>
#include <Processors/QueryPlan/LimitStep.h>
#include <Processors/QueryPlan/SourceStepWithFilter.h>
#include <Processors/QueryPlan/ObjectFilterStep.h>

namespace DB::QueryPlanOptimizations
{

void optimizePrimaryKeyConditionAndLimit(const Stack & stack)
{
    const auto & frame = stack.back();

    auto * source_step_with_filter = dynamic_cast<SourceStepWithFilterBase *>(frame.node->step.get());
    if (!source_step_with_filter)
        return;

    const auto & storage_prewhere_info = source_step_with_filter->getPrewhereInfo();
    const auto & storage_row_level_filter = source_step_with_filter->getRowLevelFilter();
    if (storage_row_level_filter)
        source_step_with_filter->addFilter(storage_row_level_filter->actions.clone(), storage_row_level_filter->column_name);
    if (storage_prewhere_info)
        source_step_with_filter->addFilter(storage_prewhere_info->prewhere_actions.clone(), storage_prewhere_info->prewhere_column_name);

    /// Collect ExpressionStep DAGs encountered while walking up the plan.
    /// When a filter references columns produced by expressions (e.g., ALIAS
    /// columns computed in "Compute alias columns" step, or renamed in
    /// "Change column names to column identifiers" step), we compose the
    /// filter through these expression DAGs so that column references are
    /// resolved to physical columns. This is essential for correct index
    /// analysis when plan optimizations like mergeExpressions have not
    /// merged these steps into the filter.
    std::vector<const ActionsDAG *> expression_dags;

    for (auto iter = stack.rbegin() + 1; iter != stack.rend(); ++iter)
    {
        if (auto * filter_step = typeid_cast<FilterStep *>(iter->node->step.get()))
        {
            auto filter_dag = filter_step->getExpression().clone();
            auto filter_column_name = filter_step->getFilterColumnName();

            /// Compose filter through accumulated expression DAGs
            /// (in bottom-to-top order). This resolves column identifiers
            /// to their underlying expressions, enabling correct index
            /// matching for ALIAS columns and renamed columns.
            for (auto it = expression_dags.rbegin(); it != expression_dags.rend(); ++it)
                filter_dag = ActionsDAG::merge((*it)->clone(), std::move(filter_dag));

            source_step_with_filter->addFilter(std::move(filter_dag), filter_column_name);
        }
        else if (auto * limit_step = typeid_cast<LimitStep *>(iter->node->step.get()))
        {
            source_step_with_filter->setLimit(limit_step->getLimitForSorting());
            break;
        }
        else if (auto * limit_by_step = typeid_cast<LimitByStep *>(iter->node->step.get()))
        {
            /// A LimitByStep whose BY-columns form a prefix of the source's sort
            /// description (i.e. applyOrder() has marked it in_order) can in
            /// principle be applied per-group at the storage layer. Push the
            /// info to the source step; it's up to the source step whether to
            /// use it. Correctness is preserved by the unaltered LimitByTransform
            /// at the top of the pipeline regardless.
            if (limit_by_step->isInOrder())
            {
                source_step_with_filter->setLimitByPushdown(LimitByPushdownInfo{
                    .columns = limit_by_step->getColumns(),
                    .length = limit_by_step->getGroupLength(),
                    .offset = limit_by_step->getGroupOffset(),
                });
            }
            break;
        }
        else if (auto * expression_step = typeid_cast<ExpressionStep *>(iter->node->step.get()))
        {
            expression_dags.push_back(&expression_step->getExpression());
            continue;
        }
        else if (auto * object_filter_step = typeid_cast<ObjectFilterStep *>(iter->node->step.get()))
        {
            source_step_with_filter->addFilter(object_filter_step->getExpression().clone(), object_filter_step->getFilterColumnName());
        }
        else
        {
            break;
        }
    }

    source_step_with_filter->applyFilters();
}

}
