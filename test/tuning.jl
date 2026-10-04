using TDAPersistenceDiagrams, MLJBase, Test, Tables
import MLJModelInterface as MMI

@testset "recommended MLJ ranges" begin
    for T in (PersistenceImageVectorizer,PersistenceCurveVectorizer,
            PersistenceLandscapeVectorizer,PersistenceDescriptorVectorizer)
        model = T()
        @test length(MMI.hyperparameter_ranges(T)) == length(fieldnames(T))
        @test !isempty(tuning_ranges(model))
        @test all(r -> r isa MLJBase.ParamRange,tuning_ranges(model))
    end
    @test length(tuning_ranges(PersistenceDescriptorVectorizer(;kind=:euler)))==1
    @test length(tuning_ranges(PersistenceDescriptorVectorizer(;kind=:block)))==2
    # No finite training bars is a legitimate CV fold, including essential H0.
    train = (h1=[PersistenceDiagram(PersistenceInterval[];dim=1),
        PersistenceDiagram(PersistenceInterval[];dim=1)],)
    model = PersistenceImageVectorizer(;width=3,height=4)
    fitted,_,_ = MMI.fit(model,0,train)
    @test size(Tables.matrix(MMI.transform(model,fitted,train))) == (2,12)
    @test all(iszero,Tables.matrix(MMI.transform(model,fitted,train)))
    heldout = (h1=[PersistenceDiagram([(0.1,0.9)];dim=1)],)
    xs,ys = copy(last(only(fitted)).xs),copy(last(only(fitted)).ys)
    MMI.transform(model,fitted,heldout)
    @test last(only(fitted)).xs==xs
    @test last(only(fitted)).ys==ys
end
